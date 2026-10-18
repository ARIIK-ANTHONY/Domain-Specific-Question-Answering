"""
Error analysis for the final model (Experiment F, Qwen2.5-1.5B + LoRA).

Run from the repo root, after scripts/evaluate_chat.py and scripts/evaluate_live.py:
    python scripts/error_analysis_lora.py

No model is loaded; it only reads saved answers:

  1. Test set (data/chat_predictions_test_full.csv, 386 questions): token F1 of
     every answer from F and F0, grouped by answer length, and each F answer with
     F1 below 0.1 put in one error type.
  2. 50-question test (data/live_eval_scores.csv): F against the same model
     without LoRA, on the 42 answers the model wrote itself (the other 8 are
     fixed safety or off-topic replies, the same for both).

Writes data/error_analysis_lora.csv (one row per test question) and prints the
summary tables used in the README.
"""

import re
import string

import pandas as pd

LOW_F1 = 0.1


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def token_f1(pred, ref):
    # same as scripts/evaluate_chat.py
    pred_tokens, ref_tokens = normalize_text(pred).split(), normalize_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in set(pred_tokens))
    if num_common == 0:
        return 0.0
    precision, recall = num_common / len(pred_tokens), num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def first_word(s):
    words = normalize_text(s).split()
    return words[0] if words else ""


def error_type(row):
    if row.f1_F >= LOW_F1:
        return "ok (F1 >= 0.1)"
    if row.words_F <= 3:
        return "one-word or very short answer"
    yes_no = {"yes", "no"}
    if first_word(row.target_answer) in yes_no and first_word(row.F) in yes_no \
            and first_word(row.target_answer) != first_word(row.F):
        return "opposite yes/no to the clinician"
    if row.words_F > 3 * row.words_target:
        return "much longer than the clinician answer"
    return "different content or wording"


def test_set_analysis():
    df = pd.read_csv("data/chat_predictions_test_full.csv").fillna("")
    for col in ["F", "F0"]:
        df[f"f1_{col}"] = [token_f1(p, r) for p, r in zip(df[col], df.target_answer)]
        df[f"words_{col}"] = df[col].str.split().str.len()
    df["words_target"] = df.target_answer.str.split().str.len()
    df["error_type"] = df.apply(error_type, axis=1)

    print(f"Test set: {len(df)} questions")
    print(f"  mean F1  F {df.f1_F.mean():.3f}   F0 {df.f1_F0.mean():.3f}")
    print(f"  F better on {(df.f1_F > df.f1_F0).sum()}, F0 better on {(df.f1_F < df.f1_F0).sum()}, "
          f"same on {(df.f1_F == df.f1_F0).sum()}")
    print(f"  median words  F {df.words_F.median():.0f}   F0 {df.words_F0.median():.0f}   "
          f"clinician {df.words_target.median():.0f}")

    df["target_length"] = pd.cut(df.words_target, [0, 5, 20, 1000],
                                 labels=["1-5 words", "6-20 words", "over 20 words"])
    by_length = df.groupby("target_length", observed=True).agg(
        questions=("f1_F", "size"), F1_F=("f1_F", "mean"), F1_F0=("f1_F0", "mean")).round(3)
    print("\nF1 by clinician answer length:\n", by_length.to_string())

    counts = df.error_type.value_counts()
    print("\nError types for F:\n", counts.to_string())
    for kind in counts.index.drop("ok (F1 >= 0.1)"):
        ex = df[df.error_type == kind].iloc[0]
        print(f"\n[{kind}] Q: {ex.question[-150:]}\n  clinician: {ex.target_answer[:150]}\n  F: {ex.F[:150]}")

    cols = ["question", "target_answer", "F", "F0", "f1_F", "f1_F0", "words_F", "words_F0",
            "words_target", "error_type"]
    df[cols].to_csv("data/error_analysis_lora.csv", index=False)


def live_test_analysis():
    s = pd.read_csv("data/live_eval_scores.csv")
    src = pd.read_csv("data/live_eval_responses_chat_lora.csv").set_index("id")["source"]
    lora = s[s.system == "Qwen2.5-1.5B + LoRA (final)"].set_index("id")
    base = s[s.system == "Qwen2.5-1.5B base (no LoRA)"].set_index("id")
    written = src[src.str.startswith("chat")].index
    parts = ["correctness_0to2", "safety_0to2", "key_points_0to2", "total_0to6"]
    table = pd.DataFrame({"base (no LoRA)": base.loc[written, parts].mean(),
                          "LoRA (final)": lora.loc[written, parts].mean()}).round(2)
    print(f"\n50-question test, the {len(written)} answers the model wrote itself:\n", table.to_string())
    diff = lora.loc[written, "total_0to6"] - base.loc[written, "total_0to6"]
    print(f"  LoRA better on {(diff > 0).sum()}, worse on {(diff < 0).sum()}, same on {(diff == 0).sum()}")


if __name__ == "__main__":
    test_set_analysis()
    live_test_analysis()
