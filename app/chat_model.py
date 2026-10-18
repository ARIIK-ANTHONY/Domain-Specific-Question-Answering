"""
Loading and running the Experiment F chat model (Qwen2.5-1.5B-Instruct + the
LoRA adapter from scripts/train_lora.py) for the app.

Unlike FLAN-T5, this is a decoder-only chat model, so it can see the recent
conversation: the last few turns are passed in as chat messages, and the
retrieved clinician answers are added to the newest user message with the
same CONTEXT_TEMPLATE used in scripts/evaluate_chat.py.
"""

import re
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from chat_prompt import BASE_CHAT_MODEL, CONTEXT_TEMPLATE, SYSTEM_PROMPT

MAX_NEW_TOKENS = 300
MAX_HISTORY_MESSAGES = 6   # last 3 user/assistant exchanges
MAX_PASSAGE_WORDS = 80     # same as scripts/evaluate_chat.py


def load(adapter_dir, device):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BASE_CHAT_MODEL)
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    # fp16 on Apple GPUs for speed at inference; fp32 on CPU / GPUs without bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if device == "mps" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE_CHAT_MODEL, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    return model, tokenizer


def build_messages(history, user_text, hits):
    """history: [{"role", "content"}, ...] of earlier turns (plain text, no
    context). hits: retrieved clinician answers for the newest message."""
    if hits:
        context = "\n".join("- " + " ".join(h["answer"].split()[:MAX_PASSAGE_WORDS]) for h in hits)
        user_text = CONTEXT_TEMPLATE.format(context=context, question=user_text)
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-MAX_HISTORY_MESSAGES:]
        + [{"role": "user", "content": user_text}]
    )


def stream_reply(model, tokenizer, messages, device):
    """Yields the reply piece by piece as it is generated."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**enc, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                  repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
    Thread(target=lambda: model.generate(**kwargs), daemon=True).start()
    # The model sometimes starts with "Answer:" (copied from the prompt format): drop it
    start = ""
    for piece in streamer:
        if start is None:
            yield piece
            continue
        start += piece
        if len(start) >= 8 or not "answer:".startswith(start.lstrip().lower()[:7]):
            yield re.sub(r"^\s*answer:\s*", "", start, flags=re.IGNORECASE)
            start = None
    if start:
        yield re.sub(r"^\s*answer:\s*", "", start, flags=re.IGNORECASE)
