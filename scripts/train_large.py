"""
Optional, NOT part of the reported experiments: fine-tune google/flan-t5-large
instead of flan-t5-base, on
the same full training set and same 20 fixed test questions as Run D, so the
two are directly comparable. This is the single most effective lever for a
genuinely stronger model without a different dataset — flan-t5-large has
roughly 3x flan-t5-base's parameters (770M vs 250M).

Run from the repo root, AFTER scripts/data_prep.py and (optionally)
scripts/train.py:
    python scripts/train_large.py

Needs more GPU memory than train.py's flan-t5-base runs — batch size is
lowered to 4 to compensate. If you hit an out-of-memory error, lower
BATCH_SIZE further (try 2) or EPOCHS.

Writes:
    data/evaluation_results_runE_large.csv   (Run D vs Run E side-by-side, if
                                               data/evaluation_results.csv exists)
    model/flan-t5-large-afrimedqa/

I didn't run this: instead of a bigger FLAN-T5, Experiment F (scripts/train_lora.py)
moved to a chat model, because the live tests showed the main problem was that
FLAN-T5 can't hold a conversation, not only its size. The app does not load this model.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # reuses load_splits, tokenize_batch, generate_answers, finetune, evaluate_predictions

# --- Configuration for this run ---
train.MODEL_NAME = "google/flan-t5-large"
EPOCHS = 3          # same as Run D, for a fair comparison
BATCH_SIZE = 4       # lower than Run D's 8 — flan-t5-large needs more memory per example
MODEL_DIR = train.REPO_ROOT / "model" / "flan-t5-large-afrimedqa"


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer

    df_train, df_val, df_test = train.load_splits()
    tokenizer = AutoTokenizer.from_pretrained(train.MODEL_NAME)

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    val_ds = Dataset.from_pandas(df_val.reset_index(drop=True))
    tokenized_train = train_ds.map(
        lambda b: train.tokenize_batch(b, tokenizer), batched=True, remove_columns=train_ds.column_names
    )
    tokenized_val = val_ds.map(
        lambda b: train.tokenize_batch(b, tokenizer), batched=True, remove_columns=val_ds.column_names
    )

    # Same fixed 20 test questions as train.py (same seed, same sampling call)
    df_test_fixed = df_test.sample(n=min(train.N_TEST_EXAMPLES, len(df_test)), random_state=train.SEED).reset_index(
        drop=True
    )
    print(f"Run E (flan-t5-large): fine-tuning on {len(tokenized_train)} examples, {EPOCHS} epochs, batch size {BATCH_SIZE}")

    model, result, state = train.finetune(
        str(train.WORK_DIR / "flan_t5_large_afrimedqa_runE"),
        tokenized_train,
        tokenized_val,
        tokenizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    df_test_fixed["runE_large_prediction"] = train.generate_answers(
        model, tokenizer, df_test_fixed["input_text"].tolist(), batch_size=BATCH_SIZE
    )

    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")
    refs = df_test_fixed["target_answer"].tolist()
    results_e = train.evaluate_predictions(df_test_fixed["runE_large_prediction"].tolist(), refs, rouge)
    print("Run E (flan-t5-large) evaluation:", results_e)

    row = {"model": "E - fine-tuned flan-t5-large (full train set, 3 epochs)", **results_e}
    results_path = train.DATA_DIR / "evaluation_results.csv"
    if results_path.exists():
        prior = pd.read_csv(results_path)
        combined = pd.concat([prior, pd.DataFrame([row])], ignore_index=True)
        combined.to_csv(train.DATA_DIR / "evaluation_results_with_runE.csv", index=False)
        print("Saved data/evaluation_results_with_runE.csv (Run A-D plus Run E, for direct comparison)")
        print(combined)
    else:
        pd.DataFrame([row]).to_csv(train.DATA_DIR / "evaluation_results_runE_large.csv", index=False)
        print("Saved data/evaluation_results_runE_large.csv (run train.py first for the full A-D comparison)")

    # Save the model (the app does not load it; see the docstring)
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    card = f"""---
language: en
license: cc-by-nc-sa-4.0
base_model: google/flan-t5-large
tags:
- medical
- question-answering
- text2text-generation
---

# FLAN-T5-large fine-tuned on AfriMed-QA v2 (Run E)

Same training data and prompt format as the flan-t5-base model (Run D), but
using `google/flan-t5-large` (~770M parameters vs ~250M) for stronger
generation quality. Trained {EPOCHS} epochs, batch size {BATCH_SIZE}, on
{len(tokenized_train)} training / {len(tokenized_val)} validation examples.

**Not a medical tool.** See the main project README for full evaluation,
error analysis, and the safety guardrail / retrieval layers the deployed
app wraps around this model.

## Evaluation (same 20 fixed test questions as Run D)
Exact Match: {results_e['exact_match']}, Token F1: {results_e['token_f1']}, ROUGE-L: {results_e['rougeL']}
"""
    with open(MODEL_DIR / "README.md", "w") as f:
        f.write(card)

    print(f"\nSaved Run E model to {MODEL_DIR}")
    print("Note: the Streamlit app does not load this model.")


if __name__ == "__main__":
    main()
