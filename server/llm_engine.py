"""Adapted PredGen speculative decoding for live ASR input.

Reuses core functions from PredGen (prompt_speculation_v2.py) and adds:
- LiveASRStreamer: drop-in replacement for InputTextStreamer, fed by real ASR partials
- live_speculative_generate(): adapted streamer_generate() for the benchmark runner
- baseline_inference(): standard sequential LLM inference for comparison
"""

import torch
import time
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


# ============================================================
# Utility functions from PredGen (reused as-is)
# ============================================================

def get_common_prefix_length(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        min_len = min(tensor1.shape[-1], tensor2.shape[-1])
        tensor1 = tensor1[..., :min_len]
        tensor2 = tensor2[..., :min_len]
    match = tensor1 == tensor2
    cumsum = match.cumprod(dim=-1)
    prefix_lengths = cumsum.sum(dim=-1)
    return prefix_lengths


def get_sd_acceptance(draft, prediction, prediction_probs, k=1):
    if draft.shape != prediction.shape:
        min_len = min(draft.shape[-1], prediction.shape[-1])
        draft = draft[..., :min_len]
        prediction = prediction[..., :min_len]
        prediction_probs = prediction_probs[:, :min_len, :]
    top_k_indices = torch.topk(prediction_probs, k, dim=-1).indices
    acceptance = (top_k_indices == draft.unsqueeze(-1)).any(dim=-1)
    rejection_mask = torch.cumprod(acceptance, dim=-1)
    accepted_count = rejection_mask.sum(-1).item()
    return accepted_count


def truncate_key_value(cache, num_of_accepted_tokens):
    """Truncate KV cache to num_of_accepted_tokens. Supports both old and new transformers API."""
    if hasattr(cache, 'crop'):
        # transformers >= 5.x
        cache.crop(num_of_accepted_tokens)
        return cache
    elif hasattr(cache, 'key_cache'):
        # transformers < 5.x (original PredGen API)
        for layer_idx in range(len(cache.key_cache)):
            cache.key_cache[layer_idx] = cache.key_cache[layer_idx][..., :num_of_accepted_tokens, :]
            cache.value_cache[layer_idx] = cache.value_cache[layer_idx][..., :num_of_accepted_tokens, :]
        return cache
    else:
        raise AttributeError(f"Unknown cache type: {type(cache)}")


def remove_last_word(s):
    words = s.split()
    if len(words) > 1:
        return " ".join(words[:-1])
    return ""


def get_sentence_break(tokenizer, keys=['.', '?', '!']):
    candidates = list([tokenizer.encode(x, add_special_tokens=False)[0] for x in keys])
    return set(candidates)


def try_to_to_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.item()
    return x


# ============================================================
# speculative_step() from PredGen (reused, simplified to topk only)
# ============================================================

@torch.no_grad()
def speculative_step(
    prev_ids, new_ids, prev_generation, model, tokenizer, past_key_values,
    max_new_token=512, prompt_text='', sentence_breaks=[], acceptance='topk',
    top_k=3, device='cuda'
):
    prev_generation = prev_generation.to(device)
    new_ids = new_ids.to(device)

    new_generation = torch.cat([new_ids, prev_generation], dim=-1)
    raw_new_generation = new_generation

    comm_prefix_len = get_common_prefix_length(new_ids, prev_ids)[0]
    if past_key_values is not None:
        if type(past_key_values) == tuple:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values = truncate_key_value(past_key_values, comm_prefix_len)
        new_generation = new_generation[:, comm_prefix_len:]

    remaining_prompt_len = new_ids.shape[-1] - comm_prefix_len

    y = model(input_ids=new_generation, past_key_values=past_key_values, use_cache=True)
    predict_next_tokens = torch.argmax(y.logits, dim=-1)

    if remaining_prompt_len > 0:
        predict_next_tokens_ans = predict_next_tokens[:, remaining_prompt_len - 1:]
    else:
        predict_next_tokens_ans = torch.cat([prev_generation[:, :1], predict_next_tokens], dim=-1)

    if remaining_prompt_len == 0:
        num_accepted_tokens = min(predict_next_tokens_ans.shape[-1], prev_generation.shape[-1])
    elif acceptance == 'greedy':
        num_accepted_tokens = get_common_prefix_length(predict_next_tokens_ans, prev_generation)[0]
    elif acceptance == 'topk':
        predict_next_tokens_probs = y.logits[:, remaining_prompt_len - 1:]
        num_accepted_tokens = get_sd_acceptance(
            prev_generation, predict_next_tokens_ans, predict_next_tokens_probs, k=top_k
        )
    else:
        raise NotImplementedError(f"Unknown acceptance={acceptance}")

    remaining_tokens = prev_generation.shape[-1] - num_accepted_tokens
    first_correct_token = predict_next_tokens_ans[:, num_accepted_tokens:num_accepted_tokens + 1]

    final_ids = torch.cat([
        new_ids,
        prev_generation[:, :num_accepted_tokens],
        first_correct_token
    ], dim=-1)

    nfe = 1
    extra_data = {}
    sentence_breaks_symbols = ['.', '?', '!']

    if remaining_tokens == 0 and first_correct_token[0][-1] == tokenizer.eos_token_id:
        generation = final_ids
        past_key_values = truncate_key_value(y.past_key_values, new_ids.shape[-1] + num_accepted_tokens)
        timestampe_first_stentence = None
        nfe_to_first_sentence = nfe
    else:
        past_key_values = truncate_key_value(y.past_key_values, new_ids.shape[-1] + num_accepted_tokens)
        timestampe_first_stentence = None
        nfe_to_first_sentence = None
        accepted_tokens = raw_new_generation[:, new_ids.shape[-1]:new_ids.shape[-1] + num_accepted_tokens + 1]
        _text = tokenizer.decode(accepted_tokens[0])
        if any([x in _text for x in sentence_breaks_symbols]):
            timestampe_first_stentence = time.time()
            nfe_to_first_sentence = nfe

        # Continue autoregressive generation
        all_tokens = predict_next_tokens_ans[:, :num_accepted_tokens + 1][0].tolist()
        for _ in range(max_new_token):
            y2 = model(input_ids=first_correct_token, use_cache=True, past_key_values=past_key_values)
            nfe += 1
            next_token = torch.argmax(y2.logits[:, -1:], dim=-1).item()
            _text = tokenizer.decode([next_token])
            if any([x in _text for x in sentence_breaks_symbols]) and timestampe_first_stentence is None:
                timestampe_first_stentence = time.time()
                nfe_to_first_sentence = nfe
            if next_token == tokenizer.eos_token_id:
                break
            all_tokens.append(next_token)
            first_correct_token[0] = next_token
        generation = torch.tensor(all_tokens)[None]

    extra_data.update(dict(
        num_accepted_tokens=try_to_to_tensor(num_accepted_tokens),
        nfe=nfe,
        nfe_to_first_sentence=nfe_to_first_sentence
    ))
    return generation, past_key_values, timestampe_first_stentence, extra_data


# ============================================================
# baseline_generate() from PredGen (reused as-is)
# ============================================================

@torch.no_grad()
def baseline_generate(new_ids, model, tokenizer, past_key_values, max_new_token):
    sentence_breaks = get_sentence_break(tokenizer)
    all_tokens_prefix = new_ids[0].tolist()
    all_tokens = []
    first_correct_token = new_ids
    timestampe_first_stentence = None
    nfe = 0
    nfe_to_first_sentence = None
    for _ in range(max_new_token):
        y2 = model(input_ids=first_correct_token, use_cache=True, past_key_values=past_key_values)
        nfe += 1
        first_correct_token = torch.argmax(y2.logits[:, -1:], dim=-1)
        next_token = first_correct_token.item()
        if next_token in sentence_breaks and timestampe_first_stentence is None:
            timestampe_first_stentence = time.time()
            nfe_to_first_sentence = nfe
        if next_token == tokenizer.eos_token_id:
            break
        all_tokens.append(next_token)
    generation = torch.tensor(all_tokens_prefix + all_tokens)[None]
    return generation[:, new_ids.shape[-1]:], past_key_values, timestampe_first_stentence, nfe_to_first_sentence


# ============================================================
# NEW: LiveASRStreamer - replaces InputTextStreamer for live ASR
# ============================================================

class LiveASRStreamer:
    """Drop-in replacement for PredGen's InputTextStreamer.

    Instead of simulating text arrival at a fixed chars/min rate,
    this accepts real ASR partial transcripts pushed via update_transcript().
    """

    def __init__(self, tokenizer, preprocessor=lambda x: x):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.current_text = ""
        self.is_final = False
        self._start_time = None
        self._speech_end_time = None
        self.dt = 0.0

    def start(self):
        self._start_time = time.time()

    def update_transcript(self, text: str, is_final: bool = False):
        """Called by ASR engine whenever a new partial transcript is available."""
        self.current_text = text
        if is_final and not self.is_final:
            self.is_final = True
            self._speech_end_time = time.time()

    def get_prompt(self, device='cuda'):
        """Same interface as InputTextStreamer.get_prompt()."""
        text = self.current_text
        if not self.is_final:
            text = remove_last_word(text)
        processed = self.preprocessor(text)
        ids = self.tokenizer(
            [processed], return_tensors='pt', add_special_tokens=False
        ).input_ids.to(device)
        return ids, text

    def advance(self, dt):
        self.dt += dt

    def is_done(self):
        return self.is_final

    @property
    def latency(self):
        """Time elapsed since ASR streaming started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time


# ============================================================
# NEW: Preprocessor for chat template
# ============================================================

def make_chat_preprocessor(tokenizer):
    """Returns a preprocessor that wraps text in the model's chat template."""
    def preprocessor(prompt):
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. The instruction given by user may be "
                "truncated. In this case, you should give answers based on your best "
                "guesses of the incomplete instruction. Do not complain about "
                "incomplete prompts."
            )},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text
    return preprocessor


# ============================================================
# NEW: live_speculative_generate() - PredGen pipeline for benchmark
# ============================================================

@torch.no_grad()
def live_speculative_generate(
    streamer: LiveASRStreamer,
    model,
    tokenizer,
    max_len=512,
    top_k=3,
):
    """Run PredGen speculative decoding driven by a LiveASRStreamer.

    The streamer should be getting update_transcript() calls from the ASR thread.
    Returns (generated_text, metrics_dict).
    """
    sentence_breaks = get_sentence_break(tokenizer)

    # Wait for initial ASR text
    while not streamer.current_text.strip():
        time.sleep(0.01)

    # Initial speculative generation
    prompt, prompt_text = streamer.get_prompt('cuda')
    t0 = time.time()
    past_key_values = DynamicCache()
    output = model.generate(
        prompt.cuda(), do_sample=False, use_cache=True,
        return_dict_in_generate=True, past_key_values=past_key_values,
        max_new_tokens=30, top_k=None, num_beams=1, repetition_penalty=1
    )
    prev_generation = output.sequences[:, prompt.shape[1]:]
    past_key_values = output.past_key_values
    prev_ids = prompt
    t1 = time.time()
    streamer.advance(t1 - t0)

    ttfs = None
    extra_data = {}
    total_nfe = 0

    while True:
        new_ids, prompt_text = streamer.get_prompt('cuda')

        if streamer.is_done():
            max_new_token = max_len
        else:
            max_new_token = 10

        t0 = time.time()
        generation, past_key_values, timestampe_first_stentence, extra_data = speculative_step(
            prev_ids, new_ids, prev_generation, model, tokenizer, past_key_values,
            max_new_token=max_new_token, prompt_text=prompt_text,
            sentence_breaks=sentence_breaks, acceptance='topk', top_k=top_k
        )
        prev_ids = new_ids
        prev_generation = generation
        t1 = time.time()

        if timestampe_first_stentence is not None and ttfs is None:
            ttfs = timestampe_first_stentence

        total_nfe += extra_data.get('nfe', 1)

        if streamer.is_done():
            streamer.advance(t1 - t0)
            break
        else:
            streamer.advance(t1 - t0)

    gen_text = tokenizer.batch_decode(generation)[0]
    if tokenizer.eos_token:
        gen_text = gen_text.replace(tokenizer.eos_token, '')

    return gen_text, {
        'ttfs': ttfs,
        'nfe': total_nfe,
        'latency': streamer.latency,
        'nfe_to_first_sentence': extra_data.get('nfe_to_first_sentence'),
    }


# ============================================================
# NEW: baseline_inference() - sequential pipeline for comparison
# ============================================================

@torch.no_grad()
def baseline_inference(text: str, model, tokenizer, preprocessor, max_len=512):
    """Standard sequential LLM inference (no speculation).

    Takes the full transcript text, processes it, and generates a response.
    Returns (generated_text, metrics_dict).
    """
    torch.cuda.synchronize()
    t0 = time.time()

    processed = preprocessor(text)
    input_ids = tokenizer(
        [processed], return_tensors='pt', add_special_tokens=False
    ).input_ids.to('cuda')

    generation, _, timestampe_first_stentence, nfe_to_first_sentence = baseline_generate(
        input_ids, model, tokenizer, DynamicCache(), max_len
    )

    gen_text = tokenizer.batch_decode(generation)[0]
    if tokenizer.eos_token:
        gen_text = gen_text.replace(tokenizer.eos_token, '')

    torch.cuda.synchronize()
    t1 = time.time()

    ttfs = None
    if timestampe_first_stentence is not None:
        ttfs = timestampe_first_stentence - t0

    return gen_text, {
        'ttfs': ttfs,
        'total_time': t1 - t0,
        'nfe_to_first_sentence': nfe_to_first_sentence,
    }


# ============================================================
# InputTextStreamer from PredGen (simulates text arriving at speech rate)
# ============================================================

class InputTextStreamer:
    """Simulates text arriving at a fixed characters-per-minute rate.

    This is PredGen's original approach: given the full transcript,
    reveal it progressively as if someone is speaking it.
    The `advance(dt)` method is called with actual compute time,
    so the "latency" metric = compute time that exceeds speech duration.
    """

    def __init__(self, prompt, tokenizer, speed=240, preprocessor=lambda x: x):
        self.speed = speed / 60  # chars per second
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.dt = 0
        self.max_time = len(self.prompt) / self.speed

    def get_prompt(self, device='cuda'):
        num_characters = int(self.speed * self.dt)
        current_prompt = self.prompt[:num_characters]
        if not self.is_done():
            current_prompt = remove_last_word(current_prompt)
        current_prompt_processed = self.preprocessor(current_prompt)
        return self.tokenizer(
            [current_prompt_processed], return_tensors='pt', add_special_tokens=False
        ).input_ids.to(device), current_prompt

    def advance(self, dt):
        self.dt += dt

    def is_done(self):
        return self.dt > self.max_time

    @property
    def latency(self):
        """Compute time that exceeded the simulated speech duration.
        Negative means compute finished before speech ended (unlikely).
        """
        return self.dt - self.max_time


# ============================================================
# predgen_speculative_generate() — PredGen's streamer_generate adapted
# ============================================================

@torch.no_grad()
def predgen_speculative_generate(
    transcript: str,
    model,
    tokenizer,
    preprocessor,
    speed: float = 600,
    max_len: int = 512,
    top_k: int = 3,
):
    """Run PredGen speculative decoding with simulated text streaming.

    Args:
        transcript: Full transcript text (pre-transcribed)
        speed: Characters per minute (simulates speech rate)

    Returns:
        (generated_text, metrics_dict) where metrics include:
        - latency: compute time after simulated speech ends (the key metric)
        - ttfs: time to first sentence
        - nfe: number of forward evaluations
    """
    sentence_breaks = get_sentence_break(tokenizer)
    streamer = InputTextStreamer(transcript, tokenizer, speed=speed, preprocessor=preprocessor)

    # Give streamer a small head start (like PredGen does with advance(5))
    # This reveals some initial text so the LLM has something to work with
    streamer.advance(0.5)

    prompt, prompt_text = streamer.get_prompt('cuda')
    t0 = time.time()
    past_key_values = DynamicCache()
    output = model.generate(
        prompt.cuda(), do_sample=False, use_cache=True,
        return_dict_in_generate=True, past_key_values=past_key_values,
        max_new_tokens=30, num_beams=1, repetition_penalty=1
    )
    prev_generation = output.sequences[:, prompt.shape[1]:]
    past_key_values = output.past_key_values
    prev_ids = prompt
    t1 = time.time()
    streamer.advance(t1 - t0)

    ttfs = None
    extra_data = {}
    total_nfe = 0

    while True:
        new_ids, prompt_text = streamer.get_prompt('cuda')

        if streamer.is_done():
            max_new_token = max_len
        else:
            max_new_token = 10

        t0 = time.time()
        generation, past_key_values, timestampe_first_stentence, extra_data = speculative_step(
            prev_ids, new_ids, prev_generation, model, tokenizer, past_key_values,
            max_new_token=max_new_token, prompt_text=prompt_text,
            sentence_breaks=sentence_breaks, acceptance='topk', top_k=top_k
        )
        prev_ids = new_ids
        prev_generation = generation
        t1 = time.time()

        if timestampe_first_stentence is not None and ttfs is None:
            ttfs = timestampe_first_stentence

        total_nfe += extra_data.get('nfe', 1)

        streamer.advance(t1 - t0)
        if streamer.is_done():
            break

    gen_text = tokenizer.batch_decode(generation)[0]
    if tokenizer.eos_token:
        gen_text = gen_text.replace(tokenizer.eos_token, '')

    return gen_text, {
        'ttfs': ttfs,
        'nfe': total_nfe,
        'latency': streamer.latency,  # KEY METRIC: compute after speech ends
        'nfe_to_first_sentence': extra_data.get('nfe_to_first_sentence'),
    }
