"""Streaming ASR engine using faster-whisper.

Supports two modes:
1. simulate_streaming() - feeds a WAV file in chunks, yielding partial transcripts with timestamps
2. transcribe_full() - one-shot transcription for baseline comparison
"""

import numpy as np
import time


class StreamingASR:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16"):
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper {model_size}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = 16000
        print("ASR model loaded.")

    def transcribe_full(self, audio: np.ndarray) -> str:
        """One-shot full transcription. Used for baseline pipeline."""
        segments, _ = self.model.transcribe(
            audio, beam_size=1, language="en", without_timestamps=True
        )
        return " ".join(s.text for s in segments).strip()

    def simulate_streaming(self, audio: np.ndarray, chunk_duration: float = 0.5):
        """Feed audio in chunks, re-transcribing each time to simulate streaming ASR.

        Yields (partial_transcript, elapsed_time) tuples.
        """
        chunk_size = int(self.sample_rate * chunk_duration)
        buffer = np.array([], dtype=np.float32)
        start_time = time.time()

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            buffer = np.concatenate([buffer, chunk])

            # Need at least 0.5s of audio for whisper
            if len(buffer) < self.sample_rate * 0.5:
                continue

            segments, _ = self.model.transcribe(
                buffer, beam_size=1, language="en", without_timestamps=True
            )
            transcript = " ".join(s.text for s in segments).strip()
            elapsed = time.time() - start_time

            if transcript:
                yield transcript, elapsed

        # Final transcription with full audio
        segments, _ = self.model.transcribe(
            buffer, beam_size=1, language="en", without_timestamps=True
        )
        final_transcript = " ".join(s.text for s in segments).strip()
        yield final_transcript, time.time() - start_time


def load_audio(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV file and return float32 numpy array at target sample rate."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != target_sr:
        # Simple resampling via linear interpolation
        duration = len(audio) / sr
        target_len = int(duration * target_sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        )
    return audio.astype(np.float32)
