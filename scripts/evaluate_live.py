"""
End-to-end evaluation of the deployed chatbot on 50 realistic user messages.

    python scripts/evaluate_live.py --backend flan
    python scripts/evaluate_live.py --backend chat --adapter model/qwen2.5-1.5b-afrimedqa-lora --name chat_lora

Why, in addition to the test-set metrics: the AfriMed-QA test questions are
mostly exam-style ("Discuss the clinical features of...") with short
clinician answers, and token F1 / ROUGE-L only measure word overlap with
them. Real users write "my body is hotting and i am vomiting". This set
(data/live_eval_questions.csv) covers what people in South Sudan actually
ask: common diseases (malaria, typhoid, cholera, kala-azar, TB, HIV, guinea
worm), maternal and child health, follow-up questions in a conversation,
spelling mistakes and informal English, health myths, emergencies, dose
requests, and out-of-domain questions. The questions were written for this
evaluation (not taken from AfriMed-QA) and are NOT used for training or
tuning anything. Each has the key points a correct answer must contain,
based on WHO / Ministry of Health guidance, so scoring is checkable.

Every message goes through the app's real pipeline (app/streamlit_app.py
prepare(): spelling, guardrails, out-of-domain check, retrieval, then the
model), and turns with the same conversation number share history, as they
would in the app. Responses are written to data/live_eval_responses_<name>.csv
for manual scoring.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
import streamlit_app as app  # noqa: E402


def load_backend(kind, adapter):
    device = app.get_device()
    if kind == "chat":
        import chat_model
        model, tokenizer = chat_model.load(adapter, device)
    else:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(app.FLAN_MODEL_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(app.FLAN_MODEL_DIR)).to(device).eval()
    return {"kind": kind, "model": model, "tokenizer": tokenizer, "device": device}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["flan", "chat"], required=True)
    parser.add_argument("--adapter", default=str(app.CHAT_ADAPTER_DIR), help="LoRA adapter folder (chat only)")
    parser.add_argument("--name", help="output suffix, defaults to the backend name")
    args = parser.parse_args()
    name = args.name or args.backend

    questions = pd.read_csv(REPO_ROOT / "data" / "live_eval_questions.csv")
    backend = load_backend(args.backend, args.adapter)
    retriever = app.Retriever(splits=app.APP_SPLITS)  # same index as the deployed app
    speller = app.load_spell_corrector(retriever, backend["tokenizer"])

    rows, history, last_question, current_conv = [], [], None, None
    for q in questions.itertuples():
        if q.conversation != current_conv:  # new conversation: forget the previous one
            history, last_question, current_conv = [], None, q.conversation
        start = time.time()
        res = app.prepare(backend, retriever, speller, q.message, history, last_question)
        if "text" in res:
            text, source = res["text"], res["source"]
        elif args.backend == "chat":
            import chat_model
            messages = chat_model.build_messages(history, res.get("corrected") or q.message, res["hits"])
            text = "".join(chat_model.stream_reply(backend["model"], backend["tokenizer"], messages,
                                                   backend["device"]))
            source = "chat_rag" if res["hits"] else "chat"
        else:
            text, source = app.flan_generate(backend, res["query"]), "model"
        # the notes the app shows under an answer (malaria test for fever, medicine warning)
        if not res.get("small_talk") and source not in ("out_of_domain", "followup"):
            notes = [n for n in (app.safety_guardrails.fever_note(res.get("corrected") or q.message),
                                 app.safety_guardrails.medication_note(res.get("query", q.message), text)) if n]
            text = text + "".join(f"\n\nNote: {n}" for n in notes)
        rows.append({
            "id": q.id, "conversation": q.conversation, "category": q.category, "message": q.message,
            "interpreted_as": res.get("corrected") or "", "source": source,
            "context_answers": len(res.get("hits", [])), "response": text.strip(),
            "seconds": round(time.time() - start, 1),
            "expected_behaviour": q.expected_behaviour, "key_points": q.key_points,
        })
        print(f"[{q.id:2d}] {source:13s} {q.message[:60]}")
        history += [{"role": "user", "content": q.message}, {"role": "assistant", "content": text}]
        if source not in ("out_of_domain", "followup", "small_talk", "guardrail") and not res.get("small_talk"):
            last_question = res.get("query", q.message)

    out = REPO_ROOT / "data" / f"live_eval_responses_{name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
