"""Benchmark runner: compares PredGen vs baseline pipeline on pre-recorded audio.

For each WAV file:
  Baseline: full ASR → full LLM generate → full TTS (sequential)
  PredGen:  streaming ASR → speculative LLM (overlapped) → TTS

Outputs metrics JSON and generated audio files for the results viewer.
"""

import argparse
import json
import os
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr_engine import StreamingASR, load_audio
from llm_engine import (
    baseline_inference,
    make_chat_preprocessor,
)
from tts_engine import StreamingTTS, save_audio


def run_baseline(wav_path, asr, model, tokenizer, tts, preprocessor, max_len=512):
    """Run the sequential baseline pipeline."""
    print(f"  [Baseline] Loading audio...")
    audio = load_audio(wav_path)
    audio_duration = len(audio) / 16000

    results = {"audio_file": os.path.basename(wav_path), "audio_duration": audio_duration}

    # Step 1: Full ASR
    print(f"  [Baseline] Running full ASR...")
    torch.cuda.synchronize()
    t_start = time.time()

    t_asr_start = time.time()
    transcript = asr.transcribe_full(audio)
    t_asr_end = time.time()
    results["asr_time"] = t_asr_end - t_asr_start
    results["transcript"] = transcript
    print(f"  [Baseline] Transcript: {transcript[:80]}...")

    # Step 2: Full LLM generation
    print(f"  [Baseline] Running LLM...")
    t_llm_start = time.time()
    gen_text, llm_metrics = baseline_inference(transcript, model, tokenizer, preprocessor, max_len)
    t_llm_end = time.time()
    results["llm_time"] = t_llm_end - t_llm_start
    results["llm_ttfs"] = llm_metrics["ttfs"]
    results["llm_nfe"] = llm_metrics.get("nfe_to_first_sentence")
    results["gen_text"] = gen_text
    print(f"  [Baseline] Response: {gen_text[:80]}...")

    # Step 3: Full TTS
    print(f"  [Baseline] Running TTS...")
    t_tts_start = time.time()
    response_audio, tts_time = tts.synthesize(gen_text)
    t_tts_end = time.time()
    results["tts_time"] = t_tts_end - t_tts_start

    torch.cuda.synchronize()
    t_end = time.time()

    results["total_time"] = t_end - t_start
    results["response_audio_duration"] = len(response_audio) / 24000 if len(response_audio) > 0 else 0

    # Save response audio
    audio_out_path = os.path.join(
        "results",
        f"baseline_{os.path.splitext(os.path.basename(wav_path))[0]}.wav"
    )
    if len(response_audio) > 0:
        save_audio(response_audio, audio_out_path)
    results["response_audio_path"] = audio_out_path

    return results


def run_predgen(wav_path, asr, model, tokenizer, tts, preprocessor, max_len=512, top_k=3, chunk_duration=0.5):
    """Run the PredGen speculative pipeline with simulated text streaming.

    Like the PredGen paper, we:
    1. Pre-transcribe the audio (get the final text)
    2. Simulate streaming by progressively revealing the transcript at speech rate
    3. The LLM speculatively generates while "waiting" for more text
    4. The key metric is "latency" = compute time that happens AFTER simulated speech ends
    """
    from llm_engine import InputTextStreamer, predgen_speculative_generate

    print(f"  [PredGen] Loading audio...")
    audio = load_audio(wav_path)
    audio_duration = len(audio) / 16000

    results = {"audio_file": os.path.basename(wav_path), "audio_duration": audio_duration}

    # Step 1: Pre-transcribe (same transcript as baseline for fair comparison)
    print(f"  [PredGen] Running full ASR...")
    t_asr_start = time.time()
    transcript = asr.transcribe_full(audio)
    t_asr_end = time.time()
    results["asr_time"] = t_asr_end - t_asr_start
    results["transcript"] = transcript
    print(f"  [PredGen] Transcript: {transcript[:80]}...")

    # Step 2: Run PredGen with simulated text streaming
    # Speed = chars/min. Audio duration determines how fast text is revealed.
    # If transcript is 100 chars and audio is 10s, speed = 100/10 * 60 = 600 chars/min
    chars_per_sec = len(transcript) / audio_duration if audio_duration > 0 else 10
    speed = chars_per_sec * 60  # chars per minute

    print(f"  [PredGen] Running speculative LLM (speed={speed:.0f} chars/min)...")
    torch.cuda.synchronize()
    t_llm_start = time.time()
    gen_text, llm_metrics = predgen_speculative_generate(
        transcript, model, tokenizer, preprocessor,
        speed=speed, max_len=max_len, top_k=top_k
    )
    torch.cuda.synchronize()
    t_llm_end = time.time()

    results["llm_time"] = t_llm_end - t_llm_start
    results["llm_ttfs"] = llm_metrics.get("ttfs")
    results["llm_nfe"] = llm_metrics.get("nfe")
    results["llm_latency"] = llm_metrics.get("latency")  # compute time after "speech ends"
    results["gen_text"] = gen_text
    print(f"  [PredGen] Response: {gen_text[:80]}...")
    print(f"  [PredGen] Speculation latency (after speech): {llm_metrics.get('latency', 0):.3f}s")

    # Step 3: TTS
    print(f"  [PredGen] Running TTS...")
    t_tts_start = time.time()
    response_audio, tts_time = tts.synthesize(gen_text)
    t_tts_end = time.time()
    results["tts_time"] = t_tts_end - t_tts_start

    # Total = ASR time + speculation latency (time after speech) + TTS
    # The speculation that happened DURING speech is "free" (overlapped)
    results["total_time"] = results["asr_time"] + llm_metrics.get("latency", 0) + results["tts_time"]
    results["total_wall_time"] = t_tts_end - t_asr_start
    results["response_audio_duration"] = len(response_audio) / 24000 if len(response_audio) > 0 else 0

    # Save response audio
    audio_out_path = os.path.join(
        "results",
        f"predgen_{os.path.splitext(os.path.basename(wav_path))[0]}.wav"
    )
    if len(response_audio) > 0:
        save_audio(response_audio, audio_out_path)
    results["response_audio_path"] = audio_out_path

    return results


def main():
    parser = argparse.ArgumentParser(description="PredGen vs Baseline Voice Agent Benchmark")
    parser.add_argument("--audio-dir", type=str, default="audio", help="Directory with WAV files")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model name")
    parser.add_argument("--whisper-model", type=str, default="large-v3", help="Whisper model size")
    parser.add_argument("--max-len", type=int, default=512, help="Max generation length")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for speculative acceptance")
    parser.add_argument("--chunk-duration", type=float, default=0.5, help="ASR chunk duration in seconds")
    parser.add_argument("--output", type=str, default="results/metrics.json", help="Output metrics file")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Find WAV files
    wav_files = sorted([
        os.path.join(args.audio_dir, f)
        for f in os.listdir(args.audio_dir)
        if f.endswith(".wav")
    ])
    if not wav_files:
        print(f"No WAV files found in {args.audio_dir}/")
        return

    print(f"Found {len(wav_files)} audio files: {[os.path.basename(f) for f in wav_files]}")

    # Load models
    print("\n=== Loading Models ===")
    asr = StreamingASR(model_size=args.whisper_model)

    print(f"Loading LLM: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("LLM loaded.")

    tts = StreamingTTS()

    preprocessor = make_chat_preprocessor(tokenizer)

    # Warmup
    print("\n=== Warmup ===")
    warmup_ids = tokenizer(["Hello"], return_tensors="pt").input_ids.to("cuda")
    _ = model.generate(warmup_ids, max_new_tokens=5, do_sample=False)
    print("Warmup done.")

    # Run benchmarks
    print(f"\n=== Running Benchmarks ===")
    all_results = []

    for wav_path in wav_files:
        print(f"\n--- {os.path.basename(wav_path)} ---")

        # Run baseline
        baseline_results = run_baseline(
            wav_path, asr, model, tokenizer, tts, preprocessor, args.max_len
        )

        # Run PredGen
        predgen_results = run_predgen(
            wav_path, asr, model, tokenizer, tts, preprocessor,
            args.max_len, args.top_k, args.chunk_duration
        )

        # Compute comparison
        comparison = {
            "audio_file": os.path.basename(wav_path),
            "audio_duration": baseline_results["audio_duration"],
            "baseline": baseline_results,
            "predgen": predgen_results,
            "speedup": baseline_results["total_time"] / predgen_results["total_time"] if predgen_results["total_time"] > 0 else 0,
            "time_saved": baseline_results["total_time"] - predgen_results["total_time"],
        }

        # TTFS comparison
        if baseline_results.get("llm_ttfs") and predgen_results.get("llm_ttfs"):
            comparison["ttfs_speedup"] = baseline_results["llm_ttfs"] / predgen_results["llm_ttfs"]
        else:
            comparison["ttfs_speedup"] = None

        all_results.append(comparison)

        print(f"\n  Baseline total: {baseline_results['total_time']:.2f}s")
        print(f"  PredGen total:  {predgen_results['total_time']:.2f}s")
        print(f"  Speedup:        {comparison['speedup']:.2f}x")
        print(f"  Time saved:     {comparison['time_saved']:.2f}s")

    # Aggregate stats
    avg_speedup = np.mean([r["speedup"] for r in all_results])
    avg_time_saved = np.mean([r["time_saved"] for r in all_results])

    summary = {
        "model": args.model,
        "whisper_model": args.whisper_model,
        "top_k": args.top_k,
        "chunk_duration": args.chunk_duration,
        "num_files": len(wav_files),
        "avg_speedup": float(avg_speedup),
        "avg_time_saved": float(avg_time_saved),
        "results": all_results,
    }

    # Save metrics
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== Results saved to {args.output} ===")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Average time saved: {avg_time_saved:.2f}s")


if __name__ == "__main__":
    main()
