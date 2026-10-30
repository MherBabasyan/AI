import sys
import time
import queue
import argparse
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
from speechbrain.inference.speaker import EncoderClassifier


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 512
SILENCE_DURATION_SEC = 0.85
ENERGY_THRESHOLD = 0.002  

audio_queue = queue.Queue()


class OnlineSpeakerDiarizer:
    """
    Online Speaker Diarization module using SpeechBrain ECAPA-TDNN embeddings
    and continuous online centroid tracking via Cosine Similarity.
    """
    def __init__(self, similarity_threshold: float = 0.68, device: str = "cpu"):
        self.device = device
        self.threshold = similarity_threshold
        self.speakers = {}        
        self.speaker_counts = {}  
        self.next_speaker_id = 1

        print(f"[INFO] Initializing ECAPA-TDNN speaker encoder on {self.device}...")
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def extract_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(tensor)
            emb_vec = embedding.squeeze().cpu().numpy()

        norm = np.linalg.norm(emb_vec)
        if norm > 0:
            emb_vec = emb_vec / norm
        return emb_vec

    def identify_speaker(self, audio_np: np.ndarray) -> str:
        # Ignore ultra-short segments (< 0.7s) to avoid polluting cluster centroids
        if len(audio_np) < 16000 * 0.7:
            if not self.speakers:
                return "Speaker 1"
            return f"Speaker {list(self.speakers.keys())[0]}"

        current_emb = self.extract_embedding(audio_np)

        if not self.speakers:
            spk_label = f"Speaker {self.next_speaker_id}"
            self.speakers[self.next_speaker_id] = current_emb
            self.speaker_counts[self.next_speaker_id] = 1
            self.next_speaker_id += 1
            return spk_label

        best_speaker_id = None
        max_similarity = -1.0

        for spk_id, centroid in self.speakers.items():
            sim = float(np.dot(current_emb, centroid))
            if sim > max_similarity:
                max_similarity = sim
                best_speaker_id = spk_id

        if max_similarity >= self.threshold:
            # Match existing speaker cluster: update running centroid
            count = self.speaker_counts[best_speaker_id]
            updated_centroid = (self.speakers[best_speaker_id] * count + current_emb) / (count + 1)
            norm = np.linalg.norm(updated_centroid)
            if norm > 0:
                updated_centroid /= norm
            self.speakers[best_speaker_id] = updated_centroid
            self.speaker_counts[best_speaker_id] += 1
            return f"Speaker {best_speaker_id}"
        else:
     
            new_id = self.next_speaker_id
            self.speakers[new_id] = current_emb
            self.speaker_counts[new_id] = 1
            self.next_speaker_id += 1
            return f"Speaker {new_id}"


def parse_args():
    parser = argparse.ArgumentParser(description="Live Armenian Transcription and English Translation")
    parser.add_argument("--mic-id", type=int, default=None, help="Input device ID for Microphone")
    parser.add_argument("--loopback-id", type=int, default=None, help="Input device ID for Loopback/System audio")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model size (tiny, base, small, medium)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--compute-type", type=str, default="int8", choices=["int8", "float16", "float32"], help="Compute quantization")
    parser.add_argument("--sim-threshold", type=float, default=0.68, help="Diarization cosine similarity threshold")
    return parser.parse_args()


def list_devices():
    print("\n=== Available Audio Devices ===")
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{idx:2d}] {dev['name']} (Max Inputs: {dev['max_input_channels']})")
    print("===============================\n")


def process_audio_worker(model: WhisperModel, diarizer: OnlineSpeakerDiarizer):
    """Worker thread that continuously consumes segmented audio chunks and performs inference."""
    while True:
        task = audio_queue.get()
        if task is None:
            break

        audio_data, source_type, start_timestamp = task

    
        if source_type == "mic":
            speaker_label = "Me"
        else:
            speaker_label = diarizer.identify_speaker(audio_data)

        try:
            print("\n[Processing speech...]", end="", flush=True)

 
            segments_hy, _ = model.transcribe(
                audio_data,
                language="hy",
                beam_size=2,
                vad_filter=True
            )
            hy_text = " ".join([s.text.strip() for s in segments_hy if s.text.strip()])

            if not hy_text:
                print("\r" + " " * 25 + "\r", end="", flush=True)
                audio_queue.task_done()
                continue

            segments_en, _ = model.transcribe(
                audio_data,
                task="translate",
                beam_size=2
            )
            en_text = " ".join([s.text.strip() for s in segments_en if s.text.strip()])


            formatted_time = time.strftime("[%H:%M:%S]", time.localtime(start_timestamp))
            print(f"\r\n{formatted_time} {speaker_label}")
            print(f"HY: {hy_text}")
            print(f"EN: {en_text}\n", flush=True)

        except Exception as err:
            print(f"\n[ERROR] Inference error: {err}", file=sys.stderr)

        audio_queue.task_done()


def create_stream(device_idx: int, source_type: str):
    """Creates a non-blocking audio input stream with energy-based VAD phrase segmentation."""
    current_chunk = []
    consecutive_silence = 0
    silence_blocks_needed = int(SAMPLE_RATE / BLOCK_SIZE * SILENCE_DURATION_SEC)
    is_speaking = False

    def callback(indata, frames, time_info, status):
        nonlocal current_chunk, consecutive_silence, is_speaking
        if status:
            print(f"[{source_type}] Buffer warning: {status}", file=sys.stderr)

        audio_np = indata.flatten().astype(np.float32)
        energy = np.sqrt(np.mean(audio_np**2))

        if energy > ENERGY_THRESHOLD:
            if not is_speaking and source_type == "mic":
                print(".", end="", flush=True)  # Ձայն լսելիս կետ է դնում
                is_speaking = True
            current_chunk.append(audio_np)
            consecutive_silence = 0
        else:
            if current_chunk:
                consecutive_silence += 1
                current_chunk.append(audio_np)
                if consecutive_silence >= silence_blocks_needed:
                    phrase = np.concatenate(current_chunk)
                    is_speaking = False
                    # Keep phrases >= 0.5s to avoid ambient clicks/taps
                    if len(phrase) >= SAMPLE_RATE * 0.5:
                        audio_queue.put((phrase, source_type, time.time() - len(phrase)/SAMPLE_RATE))
                    current_chunk = []
                    consecutive_silence = 0

    return sd.InputStream(
        device=device_idx,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=callback
    )


def main():
    args = parse_args()
    list_devices()

    mic_id = args.mic_id
    loopback_id = args.loopback_id

    if mic_id is None:
        mic_id = int(input("Enter Microphone Device Index: ").strip())
    if loopback_id is None:
        loopback_id = int(input("Enter System/Loopback Device Index: ").strip())

    print(f"\n[INFO] Loading Faster-Whisper ({args.whisper_model}) on {args.device} ({args.compute_type})...")
    whisper_model = WhisperModel(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type
    )

    diarizer = OnlineSpeakerDiarizer(
        similarity_threshold=args.sim_threshold,
        device=args.device
    )

    worker = threading.Thread(
        target=process_audio_worker,
        args=(whisper_model, diarizer),
        daemon=True
    )
    worker.start()

    print(f"\n[READY] Capturing Mic ({mic_id}) & Loopback ({loopback_id})... Speak to start.")
    print("Press Ctrl+C in terminal to stop.\n")

    with create_stream(mic_id, "mic"), create_stream(loopback_id, "loopback"):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[STOPPING] Shutting down capture streams...")
            audio_queue.put(None)
            worker.join(timeout=3)
            print("[DONE] Exited cleanly.")


if __name__ == "__main__":
    main()