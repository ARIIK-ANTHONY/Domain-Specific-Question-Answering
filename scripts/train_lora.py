"""
Experiment F: LoRA fine-tuning of a small chat model on AfriMed-QA.

Run from the repo root, after scripts/data_prep.py:
    python scripts/train_lora.py            # full run
    python scripts/train_lora.py --max-steps 5   # quick speed check

Why: Runs A-E use FLAN-T5-base, a 250M-parameter text-to-text model. Even
with retrieval (Experiment E) it can only produce one short, exam-style
reply per input; it cannot hold a conversation. Live testing of the app
showed users expect to talk to it like a doctor. Qwen2.5-1.5B-Instruct
(Qwen Team, 2024) is already trained for multi-turn chat; LoRA (Hu et al.,
2022) adapts it to AfriMed-QA by training small low-rank matrices added to
its attention and MLP layers, while the 1.5B original weights stay frozen.
That keeps training feasible on a laptop GPU (Apple M1 Pro, 16 GB) and the
saved adapter small (tens of MB instead of 3 GB).

Same data as Runs B-E: data/train.csv (917) for training, data/val.csv
(102) for validation loss / picking the best epoch. The test set is not
touched here — scripts/evaluate_chat.py evaluates the saved adapter.

Loss is computed only on the clinician's answer tokens: the system prompt
and the question are masked out (label -100), so the model learns to answer,
not to repeat questions.

Saves the best adapter (lowest validation loss) to
model/qwen2.5-1.5b-afrimedqa-lora/ and the loss curve to
data/lora_training_log.csv and figures/lora_training_loss.png.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# one GPU only: Kaggle gives two T4s, and automatic multi-GPU training breaks with LoRA +
# gradient checkpointing
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
from chat_prompt import BASE_CHAT_MODEL, SYSTEM_PROMPT  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "model" / "qwen2.5-1.5b-afrimedqa-lora"
WORK_DIR = REPO_ROOT / "notebooks" / "run_outputs" / "lora_checkpoints"
FT_PREFIX = "answer the following health question in simple language for a general audience: "

SEED = 42
MAX_LEN = 512          # question + answer tokens; longer examples are truncated
EPOCHS = 2
LEARNING_RATE = 2e-4   # usual LoRA learning rate, ~10x a full fine-tune's
BATCH_SIZE = 2
GRAD_ACCUM = 8         # effective batch size 16
LORA_R = 16            # rank of the low-rank update matrices
LORA_ALPHA = 32        # scaling: update is multiplied by alpha / r = 2
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_examples(tokenizer, df):
    """Chat-format each (question, answer) pair and mask everything but the answer."""
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    for q, a in zip(df["input_text"], df["target_answer"]):
        question = str(q).replace(FT_PREFIX, "").strip()
        prompt_msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": question}]
        prompt_ids = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        answer_ids = tokenizer(str(a).strip() + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = (list(prompt_ids) + answer_ids)[:MAX_LEN]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LEN]
        rows["input_ids"].append(ids)
        rows["attention_mask"].append([1] * len(ids))
        rows["labels"].append(labels)
    return Dataset.from_dict(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=-1, help="for a quick speed check")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    dev = device()
    print("Device:", dev)

    tokenizer = AutoTokenizer.from_pretrained(BASE_CHAT_MODEL)
    # bf16 where the GPU supports it (A100, L4). On a T4 (no bf16): weights stay in fp32 and the
    # computation runs in fp16 with automatic mixed precision, which uses the T4's tensor cores
    # and is several times faster than plain fp32; the LoRA weights and their updates stay fp32.
    # Apple M1: plain fp32 (no reliable mixed precision on MPS).
    # real bf16 needs compute capability 8+ (A100, L4). torch.cuda.is_bf16_supported() also says
    # yes on a T4 (7.5) because it can emulate bf16, and the Trainer then refuses to start.
    use_bf16 = dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    use_fp16 = dev == "cuda" and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE_CHAT_MODEL, dtype=dtype).to(dev)
    # Gradient checkpointing: recompute activations instead of storing them. Slower, but without it
    # training runs out of memory on a T4 (15 GB).
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for gradient checkpointing with frozen base weights

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=TARGET_MODULES, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_examples(tokenizer, pd.read_csv(DATA_DIR / "train.csv"))
    val_ds = build_examples(tokenizer, pd.read_csv(DATA_DIR / "val.csv"))
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=str(WORK_DIR),
        num_train_epochs=EPOCHS,
        max_steps=args.max_steps,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        lr_scheduler_type="cosine",
        warmup_steps=6,  # ~5% of the 116 optimizer steps (917 / 16 per step x 2 epochs)
        weight_decay=0.0,
        logging_steps=5,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="epoch" if args.max_steps < 0 else "no",
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    if args.max_steps > 0:
        print("Speed check done, nothing saved.")
        return

    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    log = pd.DataFrame(trainer.state.log_history)
    log.to_csv(DATA_DIR / "lora_training_log.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train_log = log.dropna(subset=["loss"]) if "loss" in log else log.iloc[0:0]
    eval_log = log.dropna(subset=["eval_loss"]) if "eval_loss" in log else log.iloc[0:0]
    plt.figure(figsize=(7, 4))
    plt.plot(train_log["epoch"], train_log["loss"], label="training loss")
    plt.plot(eval_log["epoch"], eval_log["eval_loss"], "o-", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (answer tokens only)")
    plt.title("Experiment F: LoRA fine-tuning of Qwen2.5-1.5B-Instruct")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPO_ROOT / "figures" / "lora_training_loss.png", dpi=150)
    print("Best validation loss:", trainer.state.best_metric)
    print("Saved adapter to", OUT_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
