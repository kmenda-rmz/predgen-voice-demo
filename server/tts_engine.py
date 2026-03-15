"""TTS engine using Kokoro-82M."""

import numpy as np
import time


class StreamingTTS:
    def __init__(self):
        from kokoro import KPipeline
        print("Loading Kokoro TTS...")
        self.pipeline = KPipeline(lang_code='a')  # American English
        self.sample_rate = 24000
        print("TTS model loaded.")

    def synthesize(self, text: str) -> tuple[np.ndarray, float]:
        """Synthesize text to audio.

        Returns (audio_array, generation_time_seconds).
        Audio is float32 numpy array at 24kHz.
        """
        t0 = time.time()
        audio_parts = []
        for gs, ps, audio in self.pipeline(text, voice='af_heart'):
            if audio is not None:
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                audio_parts.append(audio)

        t1 = time.time()
        if audio_parts:
            return np.concatenate(audio_parts).astype(np.float32), t1 - t0
        return np.array([], dtype=np.float32), t1 - t0


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio array to WAV file."""
    import soundfile as sf
    sf.write(path, audio, sample_rate)
