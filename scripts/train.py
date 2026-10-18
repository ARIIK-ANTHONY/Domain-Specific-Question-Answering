"""
Step 2 of the pipeline: baseline, fine-tuning experiments (B, C, D), evaluation,
and saving the final model for the Streamlit app. Mirrors Steps 6-9 and 12-13 of
notebooks/South_Sudan_Healthcare_QA.ipynb.

Run from the repo root, AFTER scripts/data_prep.py:
    python scripts/train.py

Needs a GPU: NVIDIA (CUDA) or Apple Silicon (MPS). Training on CPU is
impractically slow for this project size, so the script stops early if
neither is available.

Writes:
    data/baseline_predictions.csv, finetuned_predictions.csv, experiment_configurations.csv,
    evaluation_results.csv, training_log.csv
    figures/evaluation_metrics_comparison.png
    model/flan-t5-base-afrimedqa/   (final model + tokenizer + README.md, used by the Streamlit app)
"""

import os
import random
import re
import string
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"
MODEL_DIR = REPO_ROOT / "model" / "flan-t5-base-afrimedqa"
WORK_DIR = REPO_ROOT / "notebooks" / "run_outputs"  # checkpoints, gitignored
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
WORK_DIR.mkdir(exist_ok=True, parents=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("GPU: Apple Silicon (MPS)")
else:
    sys.exit(
        "No GPU found. This script needs an NVIDIA GPU (CUDA) or an Apple Silicon Mac (MPS). "
        "Training on CPU would take hours instead of minutes."
    )
print("Using device:", DEVICE)

ATTENTION = "eager"  # avoids a known MPS crash during beam search; identical math to fused attention

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128
N_TEST_EXAMPLES = 20
MAX_TRAIN_EXAMPLES = 800  # Run B pilot cap

os.chdir(WORK_DIR)


def free_memory():
    import gc

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def load_splits():
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (DATA_DIR / name).exists():
            sys.exit(f"Missing {DATA_DIR / name}. Run `python scripts/data_prep.py` first.")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    return df_train, df_val, df_test


def tokenize_batch(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input_text"], max_length=MAX_INPUT_LEN, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        text_target=batch["target_answer"], max_length=MAX_TARGET_LEN, truncation=True, padding="max_length"
    )
    label_ids = labels["input_ids"]
    label_ids = [[(t if t != tokenizer.pad_token_id else -100) for t in seq] for seq in label_ids]
    model_inputs["labels"] = label_ids
    return model_inputs


def generate_answers(model, tokenizer, questions, max_new_tokens=128, batch_size=8):
    model.eval()
    outputs = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LEN).to(
            DEVICE
        )
        with torch.no_grad():
            gen_ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4)
        outputs.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return outputs


def finetune(run_dir, train_dataset, tokenized_val, tokenizer, epochs, batch_size=4, grad_accum=2):
    # batch 4 x 2 accumulation steps = effective batch 8; batch 8 directly thrashed swap on a 16 GB M1 Pro
    from accelerate.state import AcceleratorState, GradientState
    from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, attn_implementation=ATTENTION).to(DEVICE)
    args = Seq2SeqTrainingArguments(
        output_dir=run_dir,
        num_train_epochs=epochs,
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        fp16=False,
        logging_steps=25,
        report_to="none",
        dataloader_pin_memory=(DEVICE == "cuda"),
        seed=SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    start = time.time()
    result = trainer.train()
    print(f"{run_dir}: trained in {time.time() - start:.1f}s, best val loss {trainer.state.best_metric:.4f}")
    trainer.save_model(f"{run_dir}/final")
    tokenizer.save_pretrained(f"{run_dir}/final")
    state = trainer.state
    del trainer
    free_memory()
    return model.eval(), result, state


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(pred, ref):
    return float(normalize_text(pred) == normalize_text(ref))


def token_f1(pred, ref):
    p, r = normalize_text(pred).split(), normalize_text(ref).split()
    if not p or not r:
        return float(p == r)
    common = {t: min(p.count(t), r.count(t)) for t in p}
    nc = sum(common.values())
    if nc == 0:
        return 0.0
    prec, rec = nc / len(p), nc / len(r)
    return 2 * prec * rec / (prec + rec)


def evaluate_predictions(preds, refs, rouge):
    em = [exact_match(p, r) for p, r in zip(preds, refs)]
    f1 = [token_f1(p, r) for p, r in zip(preds, refs)]
    rg = rouge.compute(predictions=preds, references=refs)
    return {"exact_match": float(np.mean(em)), "token_f1": float(np.mean(f1)), "rougeL": rg["rougeL"]}


def main():
    from datasets import Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    df_train, df_val, df_test = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    val_ds = Dataset.from_pandas(df_val.reset_index(drop=True))
    tokenized_train = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=val_ds.column_names)

    df_test_fixed = df_test.sample(n=min(N_TEST_EXAMPLES, len(df_test)), random_state=SEED).reset_index(drop=True)
    print(f"Fixed test subset: {len(df_test_fixed)} examples")

    # --- Step 6: baseline ---
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, attn_implementation=ATTENTION).to(DEVICE).eval()
    df_test_fixed["baseline_prediction"] = generate_answers(baseline_model, tokenizer, df_test_fixed["input_text"].tolist())
    df_test_fixed.to_csv(DATA_DIR / "baseline_predictions.csv", index=False)
    del baseline_model
    free_memory()
    print("Saved baseline_predictions.csv")

    # --- Step 7: Run B, pilot fine-tune ---
    pilot_train = tokenized_train.shuffle(seed=SEED).select(range(min(MAX_TRAIN_EXAMPLES, len(tokenized_train))))
    print(f"Run B: fine-tuning on {len(pilot_train)} examples")
    finetune_model, train_result, b_state = finetune(str(WORK_DIR / "flan_t5_afrimedqa_finetuned"), pilot_train, tokenized_val, tokenizer, epochs=2)
    df_test_fixed["finetuned_prediction"] = generate_answers(finetune_model, tokenizer, df_test_fixed["input_text"].tolist())
    df_test_fixed.to_csv(DATA_DIR / "finetuned_predictions.csv", index=False)
    del finetune_model
    free_memory()

    # --- Step 8: Run C (full train set), Run D (+1 epoch) ---
    exp_c_size = min(len(tokenized_train), MAX_TRAIN_EXAMPLES * 2)
    exp_c_train = tokenized_train.shuffle(seed=SEED).select(range(exp_c_size))
    print(f"Run C: fine-tuning on {len(exp_c_train)} examples")
    exp_c_model, exp_c_result, c_state = finetune(str(WORK_DIR / "flan_t5_afrimedqa_expC"), exp_c_train, tokenized_val, tokenizer, epochs=2)
    df_test_fixed["expC_prediction"] = generate_answers(exp_c_model, tokenizer, df_test_fixed["input_text"].tolist())
    del exp_c_model
    free_memory()

    print("Run D: fine-tuning on the same data as C, 3 epochs")
    exp_d_model, exp_d_result, d_state = finetune(str(WORK_DIR / "flan_t5_afrimedqa_expD"), exp_c_train, tokenized_val, tokenizer, epochs=3)
    df_test_fixed["expD_prediction"] = generate_answers(exp_d_model, tokenizer, df_test_fixed["input_text"].tolist())

    def eval_losses(state):
        evals = [h["eval_loss"] for h in state.log_history if "eval_loss" in h]
        return (evals[-1] if evals else None), state.best_metric

    b_last, b_best = eval_losses(b_state)
    c_last, c_best = eval_losses(c_state)
    d_last, d_best = eval_losses(d_state)

    experiment_log = pd.DataFrame(
        [
            {"experiment": "A - baseline", "train_examples": 0, "epochs": 0, "learning_rate": None, "batch_size": None,
             "runtime_sec": None, "final_train_loss": None, "final_eval_loss": None, "best_eval_loss": None},
            {"experiment": "B - finetuned pilot", "train_examples": len(pilot_train), "epochs": 2, "learning_rate": 3e-4,
             "batch_size": 8, "runtime_sec": round(train_result.metrics["train_runtime"], 1),
             "final_train_loss": train_result.training_loss, "final_eval_loss": b_last, "best_eval_loss": b_best},
            {"experiment": "C - finetuned, full train set", "train_examples": len(exp_c_train), "epochs": 2,
             "learning_rate": 3e-4, "batch_size": 8, "runtime_sec": round(exp_c_result.metrics["train_runtime"], 1),
             "final_train_loss": exp_c_result.training_loss, "final_eval_loss": c_last, "best_eval_loss": c_best},
            {"experiment": "D - finetuned, full train set, 3 epochs", "train_examples": len(exp_c_train), "epochs": 3,
             "learning_rate": 3e-4, "batch_size": 8, "runtime_sec": round(exp_d_result.metrics["train_runtime"], 1),
             "final_train_loss": exp_d_result.training_loss, "final_eval_loss": d_last, "best_eval_loss": d_best},
        ]
    )
    experiment_log.to_csv(DATA_DIR / "experiment_configurations.csv", index=False)
    print("Saved experiment_configurations.csv")

    # --- Step 9: evaluation metrics ---
    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")
    refs = df_test_fixed["target_answer"].tolist()
    results = pd.DataFrame(
        [
            {"model": "A - baseline (zero-shot)", **evaluate_predictions(df_test_fixed["baseline_prediction"].tolist(), refs, rouge)},
            {"model": "B - fine-tuned (pilot)", **evaluate_predictions(df_test_fixed["finetuned_prediction"].tolist(), refs, rouge)},
            {"model": "C - fine-tuned (full train set)", **evaluate_predictions(df_test_fixed["expC_prediction"].tolist(), refs, rouge)},
            {"model": "D - fine-tuned (full train set, 3 epochs)", **evaluate_predictions(df_test_fixed["expD_prediction"].tolist(), refs, rouge)},
        ]
    ).round(4)
    results.to_csv(DATA_DIR / "evaluation_results.csv", index=False)
    print("Saved evaluation_results.csv")
    print(results)

    import matplotlib.pyplot as plt

    x = np.arange(len(results))
    width = 0.35
    plt.figure(figsize=(7, 4))
    for i, metric in enumerate(["token_f1", "rougeL"]):
        plt.bar(x + i * width, results[metric], width, label=metric)
    plt.xticks(x + width / 2, results["model"], rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("Token F1 and ROUGE-L: baseline vs. three fine-tuned runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "evaluation_metrics_comparison.png", dpi=150)
    print("Saved figures/evaluation_metrics_comparison.png")

    def epoch_losses(state, run):
        logs = state.log_history
        rows = []
        for ev in [h for h in logs if "eval_loss" in h]:
            train = [h["loss"] for h in logs if "loss" in h and h["step"] <= ev["step"]]
            rows.append({"run": run, "epoch": round(ev["epoch"]), "train_loss": train[-1] if train else None, "val_loss": ev["eval_loss"]})
        return rows

    training_log = pd.DataFrame(epoch_losses(b_state, "B") + epoch_losses(c_state, "C") + epoch_losses(d_state, "D"))
    training_log.to_csv(DATA_DIR / "training_log.csv", index=False)
    print("Saved training_log.csv")

    df_test_fixed.to_csv(DATA_DIR / "all_predictions_fixed_test_set.csv", index=False)
    print("Saved all_predictions_fixed_test_set.csv (use this for Step 10/11 error analysis + safety scoring by hand)")

    # --- Step 13: save final model (Run D) for the Streamlit app ---
    import shutil

    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    exp_d_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    base_row, d_row = results.iloc[0], results.iloc[3]
    safety_prefix = "answer the following health question in simple language for a general audience: "
    card = f"""---
language: en
license: cc-by-nc-sa-4.0
base_model: google/flan-t5-base
tags:
- medical
- question-answering
- text2text-generation
---

# FLAN-T5-base fine-tuned on AfriMed-QA v2

A student capstone model: `google/flan-t5-base` fine-tuned to answer general health
questions, using the free-text questions in AfriMed-QA
(v2.5 CSV from https://github.com/intron-innovation/AfriMed-QA). Motivated by South Sudan's
healthcare needs, but trained on data from many African countries, not South Sudan only.

**Not a medical tool.** It often gives wrong or made-up answers. Do not use it for
diagnosis, medication doses, or emergencies.

## Training
- {len(exp_c_train)} training / {len(tokenized_val)} validation examples
- 3 epochs, learning rate 3e-4, batch size 8, weight decay 0.01, fp32
- Input format: `"{safety_prefix}" + question`
- Best validation loss: {d_best:.4f}

## Evaluation (20 fixed held-out test questions)
| Model | Exact Match | Token F1 | ROUGE-L |
|---|---|---|---|
| flan-t5-base, no fine-tuning | {base_row.exact_match} | {base_row.token_f1} | {base_row.rougeL} |
| This model | {d_row.exact_match} | {d_row.token_f1} | {d_row.rougeL} |

## License
Training data (AfriMed-QA, GitHub copy) is CC BY-NC-SA 4.0; this model uses the same license.
"""
    with open(MODEL_DIR / "README.md", "w") as f:
        f.write(card)

    print(f"\nFinal model saved to {MODEL_DIR}")
    print("Run the Streamlit app now with: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
