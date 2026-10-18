"""
Experiment E: retrieval-augmented generation (RAG) with the fine-tuned model.

Run from the repo root, after scripts/train.py (needs model/flan-t5-base-afrimedqa/):
    python scripts/evaluate_rag.py

Research question: Run D only ever sees the question, so everything it says
has to come from what it memorised from 917 training examples — and Step 10's
error analysis found hallucination was its biggest failure. Does giving the
same fine-tuned model the most similar clinician-written answers as context
make its answers more accurate?

Procedure (the test set is never used to choose anything):
  1. Tuning, on the 102 validation questions, with a retrieval index built
     from the TRAINING split only (so a question can't retrieve itself):
     try both prompt formats in app/rag.py x k = 1, 3, 5 retrieved answers,
     keep the configuration with the best token F1.
  2. Testing, with the index built from train + val (what the app uses).
     Six systems, each on the notebook's 20 fixed test questions (same
     sample as Runs A-D) and on the full 386-question test set:
       A      zero-shot flan-t5-base (the notebook's baseline)
       D      Run D, notebook decoding (reproduces the notebook's number)
       D+dec  Run D + repetition penalty decoding fix (what the app used before)
       R      retrieval only: show the nearest clinician answer, no generation
       E-base RAG with flan-t5-base, NOT fine-tuned (is fine-tuning still needed?)
       E      RAG with Run D (the proposed system)

Writes data/rag_val_tuning.csv, data/rag_experiment_results.csv,
data/rag_predictions_test20.csv, data/rag_predictions_test_full.csv and
figures/rag_experiment_comparison.png.
"""

import os
import re
import string
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
import rag  # noqa: E402
from retrieval import Retriever  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
FIG_DIR = REPO_ROOT / "figures"
BASE_MODEL = "google/flan-t5-base"
FINETUNED_DIR = REPO_ROOT / "model" / "flan-t5-base-afrimedqa"
SEED = 42
N_TEST_FIXED = 20
MAX_INPUT_LEN = 256  # same as the notebook, for prompts without context
MAX_NEW_TOKENS = 128
BATCH_SIZE = 8

NOTEBOOK_DECODING = dict(num_beams=4)
APP_DECODING = dict(num_beams=4, repetition_penalty=1.3, no_repeat_ngram_size=3)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ---- metrics: identical to the notebook's Step 9 -----------------------------
rouge = evaluate.load("rouge")


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred, ref):
    pred_tokens, ref_tokens = normalize_text(pred).split(), normalize_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens))
    if num_common == 0:
        return 0.0
    precision, recall = num_common / len(pred_tokens), num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(preds, refs):
    return {
        "exact_match": float(np.mean([normalize_text(p) == normalize_text(r) for p, r in zip(preds, refs)])),
        "token_f1": float(np.mean([token_f1(p, r) for p, r in zip(preds, refs)])),
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
    }


# ---- generation ---------------------------------------------------------------
def load(path):
    # eager attention: the fused kernel crashes on Apple GPUs during beam search (see notebook Step 2)
    return AutoModelForSeq2SeqLM.from_pretrained(str(path), attn_implementation="eager").to(DEVICE).eval()


def generate(model, tokenizer, inputs, decoding, max_input_len):
    outputs = []
    for i in range(0, len(inputs), BATCH_SIZE):
        enc = tokenizer(inputs[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_input_len).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, **decoding)
        outputs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
        print(f"    {min(i + BATCH_SIZE, len(inputs))}/{len(inputs)}", end="\r", flush=True)
    print()
    return outputs


def rag_inputs(retriever, questions, fmt, k):
    return [rag.build_input(q, retriever.search(q, k=k), fmt) for q in questions]


def main():
    print("Device:", DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_DIR))
    run_d = load(FINETUNED_DIR)

    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    # the notebook's exact 20-question subset (Step 6)
    test_fixed_idx = test.sample(n=N_TEST_FIXED, random_state=SEED).index

    # ---- 1. tune format and k on validation (train-only index) ----
    print("\n[1/2] Tuning on validation set")
    train_index = Retriever(splits=("train.csv",))
    tuning = []
    for fmt in rag.PROMPT_FORMATS:
        for k in (1, 3, 5):
            print(f"  format={fmt} k={k}")
            preds = generate(run_d, tokenizer, rag_inputs(train_index, val["input_text"].tolist(), fmt, k),
                             APP_DECODING, rag.MAX_RAG_INPUT_LEN)
            tuning.append({"prompt_format": fmt, "k": k, **evaluate_predictions(preds, val["target_answer"].tolist())})
    tuning_df = pd.DataFrame(tuning).round(4)
    tuning_df.to_csv(DATA_DIR / "rag_val_tuning.csv", index=False)
    print(tuning_df.to_string(index=False))
    best = tuning_df.sort_values(["token_f1", "rougeL"], ascending=False).iloc[0]
    best_fmt, best_k = best["prompt_format"], int(best["k"])
    print(f"  -> chosen on validation: format={best_fmt}, k={best_k}")
    if (best_fmt, best_k) != (rag.DEFAULT_FORMAT, rag.DEFAULT_K):
        print(f"  NOTE: update DEFAULT_FORMAT/DEFAULT_K in app/rag.py to {best_fmt!r}, {best_k}")

    # ---- 2. test (train + val index, as in the app) ----
    print("\n[2/2] Test set")
    index = Retriever(splits=("train.csv", "val.csv"), encoder=train_index.encoder)
    questions = test["input_text"].tolist()
    preds = pd.DataFrame({"input_text": questions, "target_answer": test["target_answer"]})

    print("  D: Run D, notebook decoding")
    preds["D"] = generate(run_d, tokenizer, questions, NOTEBOOK_DECODING, MAX_INPUT_LEN)
    print("  D+dec: Run D, app decoding")
    preds["D+dec"] = generate(run_d, tokenizer, questions, APP_DECODING, MAX_INPUT_LEN)
    print("  E: RAG with Run D")
    contexts = rag_inputs(index, questions, best_fmt, best_k)
    preds["E"] = generate(run_d, tokenizer, contexts, APP_DECODING, rag.MAX_RAG_INPUT_LEN)
    print("  R: retrieval only")
    preds["R"] = [index.search(q, k=1)[0]["answer"] for q in questions]
    preds["top1_similarity"] = [index.search(q, k=1)[0]["similarity"] for q in questions]
    del run_d

    base = load(BASE_MODEL)
    print("  A: zero-shot baseline")
    preds["A"] = generate(base, tokenizer, questions, NOTEBOOK_DECODING, MAX_INPUT_LEN)
    print("  E-base: RAG with flan-t5-base, not fine-tuned")
    preds["E-base"] = generate(base, tokenizer, contexts, APP_DECODING, rag.MAX_RAG_INPUT_LEN)
    del base

    systems = {
        "A": "A - zero-shot flan-t5-base (baseline)",
        "D": "D - fine-tuned, 3 epochs (notebook decoding)",
        "D+dec": "D + repetition-penalty decoding",
        "R": "R - retrieval only (nearest clinician answer)",
        "E-base": f"E-base - RAG, k={best_k}, NOT fine-tuned",
        "E": f"E - RAG, k={best_k}, fine-tuned Run D",
    }
    rows = []
    for subset_name, subset in (("fixed 20 (as Runs A-D)", preds.loc[test_fixed_idx]),
                                ("full test set (386)", preds)):
        for col, label in systems.items():
            rows.append({"test_subset": subset_name, "system": label,
                         **evaluate_predictions(subset[col].tolist(), subset["target_answer"].tolist())})
    results = pd.DataFrame(rows).round(4)
    results.to_csv(DATA_DIR / "rag_experiment_results.csv", index=False)
    preds.loc[test_fixed_idx].reset_index(drop=True).to_csv(DATA_DIR / "rag_predictions_test20.csv", index=False)
    preds.to_csv(DATA_DIR / "rag_predictions_test_full.csv", index=False)
    print(results.to_string(index=False))

    full = results[results.test_subset.str.startswith("full")]
    x = np.arange(len(full))
    plt.figure(figsize=(8, 4.5))
    for i, metric in enumerate(["token_f1", "rougeL"]):
        plt.bar(x + i * 0.38, full[metric], 0.38, label=metric)
    plt.xticks(x + 0.19, list(systems), rotation=0)
    plt.ylabel("Score")
    plt.title("Full test set (386 questions): baseline, fine-tuning, retrieval and RAG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rag_experiment_comparison.png", dpi=150)
    print("\nSaved data/rag_*.csv and figures/rag_experiment_comparison.png")


if __name__ == "__main__":
    main()
