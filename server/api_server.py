"""FastAPI server for live PredGen vs Baseline demo.

Loads all models at startup, exposes endpoints for running inference
on uploaded or pre-recorded audio files.
"""

import base64
import io
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr_engine import StreamingASR, load_audio
from llm_engine import baseline_inference, make_chat_preprocessor
from run_benchmark import run_baseline, run_predgen
from tts_engine import StreamingTTS, save_audio

app = FastAPI(title="PredGen Voice Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
asr = model = tokenizer = tts = preprocessor = None
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "audio")


@app.on_event("startup")
def load_models():
    global asr, model, tokenizer, tts, preprocessor

    print("=== Loading Models ===")
    asr = StreamingASR(model_size="large-v3")

    print("Loading LLM: Qwen/Qwen2.5-14B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct", device_map="cuda", dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("LLM loaded.")

    tts = StreamingTTS()
    preprocessor = make_chat_preprocessor(tokenizer)

    # Warmup
    print("Warming up...")
    warmup_ids = tokenizer(["Hello"], return_tensors="pt").input_ids.to("cuda")
    _ = model.generate(warmup_ids, max_new_tokens=5, do_sample=False)
    print("=== Server ready ===")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": model is not None}


@app.get("/samples")
def list_samples():
    audio_dir = os.path.abspath(AUDIO_DIR)
    if not os.path.isdir(audio_dir):
        return {"samples": []}
    files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))
    return {"samples": files}


@app.post("/run")
def run_inference(
    mode: str = Form(...),
    sample: str = Form(None),
    file: UploadFile = File(None),
):
    # Resolve audio to a WAV file path
    tmp_file = None
    try:
        if sample:
            wav_path = os.path.abspath(os.path.join(AUDIO_DIR, sample))
            if not os.path.isfile(wav_path):
                return {"error": f"Sample not found: {sample}"}
        elif file:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file.write(file.file.read())
            tmp_file.close()
            wav_path = tmp_file.name
        else:
            return {"error": "Provide either 'sample' or 'file'"}

        # Run the selected pipeline
        if mode == "baseline":
            results = run_baseline(wav_path, asr, model, tokenizer, tts, preprocessor)
        elif mode == "predgen":
            results = run_predgen(wav_path, asr, model, tokenizer, tts, preprocessor)
        else:
            return {"error": f"Invalid mode: {mode}. Use 'baseline' or 'predgen'"}

        # Base64 encode response audio if it exists
        audio_b64 = None
        audio_path = results.get("response_audio_path")
        if audio_path and os.path.isfile(audio_path):
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "mode": mode,
            "audio_file": results.get("audio_file"),
            "audio_duration": results.get("audio_duration"),
            "transcript": results.get("transcript"),
            "gen_text": results.get("gen_text"),
            "asr_time": results.get("asr_time"),
            "llm_time": results.get("llm_time"),
            "llm_ttfs": results.get("llm_ttfs"),
            "tts_time": results.get("tts_time"),
            "total_time": results.get("total_time"),
            "llm_latency": results.get("llm_latency"),
            "response_audio_b64": audio_b64,
        }
    finally:
        if tmp_file and os.path.isfile(tmp_file.name):
            os.unlink(tmp_file.name)
