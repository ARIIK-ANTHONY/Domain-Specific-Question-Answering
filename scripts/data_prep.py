"""
Step 1 of the pipeline: load AfriMed-QA v2, clean it, and split it into
train/val/test. This mirrors Steps 3-5 of the original notebook
(notebooks/South_Sudan_Healthcare_QA.ipynb) exactly, so the numbers reported
in the report/README match what this script produces.

Run from the repo root:
    python scripts/data_prep.py

Writes:
    data/train.csv, data/val.csv, data/test.csv   (the actual splits used for training/eval)
    data/data_pipeline_counts.csv                 (row counts at each cleaning step)
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# The authors' public GitHub copy, pinned to one commit so the data can't change under us.
DATA_COMMIT = "07850ded9dbf918c0f9d31450f8eec7e24b3608c"
DATA_URL = (
    "https://raw.githubusercontent.com/intron-innovation/AfriMed-QA/"
    f"{DATA_COMMIT}/data/afri_med_qa_15k_v2.5_phase_2_15275.csv"
)


def build_input(row, question_col):
    prefix = "answer the following health question in simple language for a general audience: "
    q = str(row.get(question_col, "") or "").strip()
    prompt_text = str(row.get("prompt", "") or "").strip()
    if prompt_text and prompt_text.lower() != "nan":
        return f"{prefix}context: {prompt_text} question: {q}"
    return f"{prefix}{q}"


def pick_answer(row):
    for col in ("answer_rationale", "correct_answer"):
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col].strip()
    return None


def main():
    print(f"Loading {DATA_URL} ...")
    df_raw = pd.read_csv(DATA_URL)
    print(f"Raw rows: {len(df_raw)}")

    if "question_type" not in df_raw.columns:
        sys.exit("No 'question_type' column found — the dataset format changed, stop and inspect it.")

    # Keep only free-form question types (drop MCQ types)
    generative_types = [t for t in df_raw["question_type"].unique() if "mcq" not in str(t).lower()]
    df_gen = df_raw[df_raw["question_type"].isin(generative_types)].copy()
    n_before = len(df_gen)
    print(f"After dropping MCQ types, keeping {generative_types}: {n_before}")

    question_col = "question_clean" if "question_clean" in df_gen.columns else "question"

    df_gen["target_answer"] = df_gen.apply(pick_answer, axis=1)
    df_gen["input_text"] = df_gen.apply(lambda r: build_input(r, question_col), axis=1)

    df_gen = df_gen[df_gen["target_answer"].notna() & (df_gen["input_text"].str.len() > 0)]
    n_after_nonempty = len(df_gen)
    print(f"After dropping rows with no written answer: {n_after_nonempty}")

    n_before_dedup = len(df_gen)
    df_gen = df_gen.drop_duplicates(subset=[question_col])
    n_after_dedup = len(df_gen)
    print(f"After de-duplicating on '{question_col}': {n_after_dedup} (dropped {n_before_dedup - n_after_dedup})")

    # Official train/test split from the dataset card, with our own val carved from train
    if "split" in df_gen.columns and df_gen["split"].notna().any():
        print("Using the dataset's official `split` column.")
        df_train_full = df_gen[df_gen["split"].str.lower() == "train"].copy()
        df_test = df_gen[df_gen["split"].str.lower() != "train"].copy()
    else:
        print("No usable official split column found — falling back to a 90/10 split (seed fixed).")
        df_train_full, df_test = train_test_split(df_gen, test_size=0.10, random_state=SEED)

    df_train, df_val = train_test_split(df_train_full, test_size=0.10, random_state=SEED)

    print(f"Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    # Leakage check
    train_qs = set(df_train[question_col])
    val_qs = set(df_val[question_col])
    test_qs = set(df_test[question_col])
    assert not (train_qs & test_qs), "Leakage detected between train and test."
    assert not (val_qs & test_qs), "Leakage detected between val and test."
    print("No leakage detected between splits.")

    cols_to_keep = ["input_text", "target_answer"]
    df_train[cols_to_keep].to_csv(DATA_DIR / "train.csv", index=False)
    df_val[cols_to_keep].to_csv(DATA_DIR / "val.csv", index=False)
    df_test[cols_to_keep].to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Saved train.csv / val.csv / test.csv to {DATA_DIR}")

    pipeline_counts = pd.DataFrame(
        [
            {"stage": "Raw dataset", "rows": len(df_raw)},
            {"stage": "Drop MCQ types", "rows": n_before},
            {"stage": "Drop rows with no written answer", "rows": n_after_nonempty},
            {"stage": "De-duplicate on question text", "rows": n_after_dedup},
            {"stage": "Train", "rows": len(df_train)},
            {"stage": "Validation", "rows": len(df_val)},
            {"stage": "Test", "rows": len(df_test)},
        ]
    )
    pipeline_counts.to_csv(DATA_DIR / "data_pipeline_counts.csv", index=False)
    print("Saved data_pipeline_counts.csv")
    print(pipeline_counts)


if __name__ == "__main__":
    main()
