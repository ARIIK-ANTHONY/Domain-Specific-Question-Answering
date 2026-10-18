"""
Streamlit chatbot for the AfriMed-QA healthcare assistant.

It does not train anything. It loads one of the fine-tuned models:

  - Chat mode (preferred): Qwen2.5-1.5B-Instruct + the LoRA adapter from
    scripts/train_lora.py (Experiment F), in model/qwen2.5-1.5b-afrimedqa-lora/.
    A multi-turn chat model: it sees the recent conversation and is given
    the most similar clinician answers from AfriMed-QA as context.
  - Fallback: FLAN-T5-base fine-tuned by scripts/train.py (Run D), in
    model/flan-t5-base-afrimedqa/. Single-turn, answers one question at a time.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safety_guardrails  # noqa: E402
from query_cleanup import (FOLLOWUP_RESPONSE, SpellCorrector, resolve_followup,  # noqa: E402
                           small_talk_reply)
from retrieval import DIRECT_ANSWER_THRESHOLD, Retriever  # noqa: E402

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAT_ADAPTER_DIR = REPO_ROOT / "model" / "qwen2.5-1.5b-afrimedqa-lora"
FLAN_MODEL_DIR = REPO_ROOT / "model" / "flan-t5-base-afrimedqa"
PREFIX = "answer the following health question in simple language for a general audience: "
FLAN_MAX_INPUT_LEN = 256
FLAN_MAX_NEW_TOKENS = 128

# Calibrated on example questions: off-topic ones ("weather in Juba", "how do
# I cook rice") scored 0.16-0.33 against the closest AfriMed-QA question,
# everyday health questions 0.47 and up.
OUT_OF_DOMAIN_THRESHOLD = 0.40
# Similarity alone refused 9 of 48 real health questions in the 50-question live
# evaluation (scripts/evaluate_live.py): "what is kala azar" scored 0.28, below
# "can you help me write a CV" (0.27 vs 0.28), because the embedding model
# doesn't know local disease names and informal phrasing ("my body is hotting")
# lowers similarity. So a message is only out of domain if it is dissimilar AND
# contains none of these health words.
HEALTH_TERMS = re.compile(
    r"\b(sick|ill|illness|pain|painful|hurt|ache|fever|hot body|hotting|vomit|diarr|cough|blood|bleed"
    r"|rash|itch|swell|swollen|sore|weak|tired|sleep|sad|stress|depress|anxi|mental|trauma"
    r"|pregnan|antenatal|prenatal|baby|babies|infant|newborn|child|children|son|daughter|breastf"
    r"|birth|deliver|period|menstru|malaria|typhoid|cholera|kala|azar|tb|tuberculosis|hiv|aids"
    r"|hepatitis|measles|polio|vaccin|immuni|worm|deworm|ors|rehydrat|dehydrat|medicine|medication"
    r"|drug|tablet|pill|doctor|nurse|clinic|hospital|health|disease|infect|virus|bacteria|germ"
    r"|wash|soap|hygien|latrine|toilet|nutrition|malnutrition|diet|diabet|pressure|heart|stomach"
    r"|belly|head|eye|skin|tooth|teeth|ear|chest|breath|snake|bite|burn|wound|injur|poison"
    r"|swallow|condom|family planning|contracep)\w*", re.IGNORECASE)
NEW_QUESTION = re.compile(r"(what|who|whom|where|when|which|why|how|can|could|will|would|is|are|do|does"
                          r"|write|tell|give|show|help|explain)\b", re.IGNORECASE)
OUT_OF_DOMAIN_RESPONSE = (
    "That doesn't look like a health question, and I can only help with health topics "
    "(for example malaria, typhoid, diabetes, pregnancy care or nutrition). What health "
    "question can I help you with?"
)
# Retrieved answers are only given to the chat model as context above this
# similarity. Experiment E found that weakly related context made answers
# worse (data/rag_predictions_test_full.csv: context only helped for test
# questions whose best match was >= 0.7), and in live testing FLAN-T5 copied a
# 53%-similar "sleeping sickness" answer for a flu + headache question.
MIN_CONTEXT_SIMILARITY = 0.55
# The deployed app searches all 1,405 clinician-answered questions, test split
# included, so users get the whole dataset. This is only done in the app: every
# reported score comes from scripts that index train (+ val) only, so test
# questions never see their own answers during evaluation.
APP_SPLITS = ("train.csv", "val.csv", "test.csv")
CHAT_RAG_K = 3  # updated from data/chat_val_tuning.csv after Experiment F

DISCLAIMER = (
    "This assistant gives general health information only. It is not a substitute for "
    "a health worker. For urgent symptoms, go to the nearest health facility or call "
    "emergency services."
)

st.set_page_config(page_title="Health Assistant for South Sudan", layout="centered")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner="Loading the fine-tuned model (first time can take a minute)...")
def load_backend():
    device = get_device()
    if CHAT_ADAPTER_DIR.exists():
        import chat_model
        model, tokenizer = chat_model.load(CHAT_ADAPTER_DIR, device)
        return {"kind": "chat", "model": model, "tokenizer": tokenizer, "device": device,
                "name": "Qwen2.5-1.5B-Instruct + LoRA (AfriMed-QA)"}
    if FLAN_MODEL_DIR.exists():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(FLAN_MODEL_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(FLAN_MODEL_DIR)).to(device).eval()
        return {"kind": "flan", "model": model, "tokenizer": tokenizer, "device": device,
                "name": "FLAN-T5-base, fine-tuned (Run D)"}
    return None


@st.cache_resource(show_spinner="Indexing clinician answers...")
def load_retriever():
    return Retriever(splits=APP_SPLITS)


@st.cache_resource(show_spinner=False)
def load_spell_corrector(_retriever, _tokenizer):
    return SpellCorrector(_retriever.questions + _retriever.answers,
                          known_words=SpellCorrector.words_from_tokenizer(_tokenizer))


def flan_generate(backend, question: str) -> str:
    """Run D with the repetition-penalty decoding fix: the best FLAN-T5 system on
    the full test set (Experiment E: token F1 0.093, vs 0.084 with retrieved context)."""
    tok, model = backend["tokenizer"], backend["model"]
    enc = tok(PREFIX + question, return_tensors="pt", truncation=True,
              max_length=FLAN_MAX_INPUT_LEN).to(backend["device"])
    with torch.no_grad():
        ids = model.generate(**enc, max_new_tokens=FLAN_MAX_NEW_TOKENS, num_beams=4,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    return tok.batch_decode(ids, skip_special_tokens=True)[0]


def prepare(backend, retriever, speller, message: str, history, last_question):
    """Everything that happens before the model writes: clean-up, safety
    checks, out-of-domain check and retrieval. Returns a dict; if it has a
    "text", that fixed reply is shown and the model is not called."""
    chat = backend["kind"] == "chat"

    if small_talk_reply(message):
        # the chat model handles greetings and thanks naturally; FLAN-T5 can't
        if not chat:
            return {"text": small_talk_reply(message), "source": "small_talk"}
        return {"query": message, "hits": [], "small_talk": True}

    if chat:
        # the chat model sees the conversation itself; the previous question is
        # only added to the retrieval query, so short follow-ups find the topic
        query, followup_kind = message, None
        if last_question and len(message.split()) <= 8:
            query = f"{message} ({last_question})"
    else:
        query, followup_kind = resolve_followup(message, last_question)
        if followup_kind == "no_context":
            return {"text": FOLLOWUP_RESPONSE, "source": "followup"}

    corrected = speller.correct(query) if speller else query
    corrected_message = speller.correct(message) if speller else message
    info = {"query": corrected,
            "corrected": corrected_message if corrected_message != message else None}

    # chat mode: check the new message itself, not the previous topic appended to it
    guard = safety_guardrails.check(corrected_message if chat else corrected, include_diagnosis=not chat)
    if guard:
        return {**info, "text": guard, "source": "guardrail"}

    hits = retriever.search(corrected, k=CHAT_RAG_K) if retriever.ready else []
    best = hits[0]["similarity"] if hits else 0.0
    # out of domain: short replies mid-conversation ("yes, for 3 days") are exempt,
    # since on their own they don't look like health questions
    # Judged on the new message alone: in chat mode the retrieval query has the
    # previous topic appended, which let "what is the capital of South Sudan"
    # through after a malaria question.
    # A short statement mid-conversation ("he is 3 years old", "yes, since yesterday")
    # answers the conversation; a short new question ("what is the capital of
    # South Sudan") is still checked.
    short_reply = bool(chat and history and len(message.split()) <= 8
                       and not NEW_QUESTION.match(message.strip()))
    own_hits = retriever.search(corrected_message, k=1) if (retriever.ready and corrected != corrected_message) else hits
    own_best = own_hits[0]["similarity"] if own_hits else 0.0
    if own_hits and own_best < OUT_OF_DOMAIN_THRESHOLD and not short_reply \
            and not HEALTH_TERMS.search(corrected_message):
        return {**info, "text": OUT_OF_DOMAIN_RESPONSE, "source": "out_of_domain", "similarity": own_best}

    if not chat and best >= DIRECT_ANSWER_THRESHOLD and followup_kind != "more":
        return {**info, "text": hits[0]["answer"], "source": "retrieval", "similarity": best,
                "sources": hits[:1]}

    return {**info, "hits": [h for h in hits if h["similarity"] >= MIN_CONTEXT_SIMILARITY],
            "related": hits}


def main():
    st.title("Health Assistant for South Sudan")
    st.info(DISCLAIMER)

    backend = load_backend()
    retriever = load_retriever()
    if backend is None:
        st.error(
            "No trained model found in `model/`. Train one first (see README):\n\n"
            "```\npython scripts/data_prep.py\npython scripts/train.py        # FLAN-T5 (Run D)\n"
            "python scripts/train_lora.py   # chat model (Experiment F)\n```"
        )
        st.stop()
    chat = backend["kind"] == "chat"
    speller = load_spell_corrector(retriever, backend["tokenizer"]) if retriever.ready else None
    st.caption(f"Model: {backend['name']}, running on {backend['device']}")

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.last_question = None

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    message = st.chat_input("Ask a health question...")
    if message:
        with st.chat_message("user"):
            st.markdown(message)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = prepare(backend, retriever, speller, message,
                                 st.session_state.history, st.session_state.last_question)
            if result.get("corrected"):
                st.caption(f"Interpreted as: *{result['corrected']}*")

            if "text" in result:
                text = result["text"]
                st.markdown(text)
            elif chat:
                import chat_model
                messages = chat_model.build_messages(st.session_state.history, result.get("corrected") or message,
                                                     result["hits"])
                text = st.write_stream(chat_model.stream_reply(
                    backend["model"], backend["tokenizer"], messages, backend["device"]))
                result["source"] = "chat_rag" if result["hits"] else "chat"
            else:
                with st.spinner("Thinking..."):
                    text = flan_generate(backend, result["query"])
                st.markdown(text)
                result["source"] = "model"

            notes = []
            if not result.get("small_talk") and result.get("source") not in ("out_of_domain", "followup"):
                # fever note on the new message only, so it isn't repeated on every follow-up
                notes = [n for n in (safety_guardrails.fever_note(result.get("corrected") or message),
                                     safety_guardrails.medication_note(result.get("query", message), text)) if n]
            for note in notes:  # part of the same reply, as its last line
                st.markdown(f"**Safety note:** {note}")
            labels = {
                "small_talk": "Fixed reply (not model-generated)",
                "followup": "Follow-up with no earlier question to refer to (not model-generated)",
                "guardrail": "Safety guardrail (not model-generated)",
                "out_of_domain": f"Outside the health domain (closest AfriMed-QA question only "
                                 f"{result.get('similarity', 0):.0%} similar), not model-generated",
                "retrieval": f"Near-identical question on file ({result.get('similarity', 0):.0%} similar): "
                             f"clinician-written answer, not model-generated",
                "chat_rag": "Written by the fine-tuned chat model, using similar clinician answers as context",
                "chat": "Written by the fine-tuned chat model",
                "model": "Written by the fine-tuned FLAN-T5 model",
            }
            if not result.get("small_talk"):
                st.caption(labels[result["source"]])

        st.session_state.history.append({"role": "user", "content": message})
        st.session_state.history.append(
            {"role": "assistant", "content": text + "".join(f"\n\nNote: {n}" for n in notes)})
        if result.get("source") not in ("out_of_domain", "followup", "small_talk", "guardrail") \
                and not result.get("small_talk"):
            st.session_state.last_question = result.get("query", message)

    with st.sidebar:
        st.header("About")
        st.write(
            "A student capstone project (ALU NLP Summative). The model is fine-tuned on the "
            "free-text questions in [AfriMed-QA v2](https://github.com/intron-innovation/AfriMed-QA), "
            "a pan-African dataset of health questions answered by clinicians."
        )
        if chat:
            st.write(
                "**How a reply is produced:**\n"
                "1. Spelling is corrected, and short follow-ups are linked to the topic "
                "you were discussing.\n"
                "2. Emergencies, medicine doses and medicine safety in pregnancy or for "
                "infants get a fixed safety reply instead of a model answer.\n"
                "3. Questions with nothing similar in the health dataset are declined.\n"
                "4. The most similar clinician answers are found with sentence embeddings "
                "(all-MiniLM-L6-v2) and given to the model as context.\n"
                "5. Qwen2.5-1.5B-Instruct, adapted to AfriMed-QA with LoRA, writes the "
                "reply, seeing the last few turns of the conversation."
            )
        else:
            st.write(
                "**How an answer is produced:**\n"
                "1. Spelling is corrected and follow-ups are linked to your previous question.\n"
                "2. Emergencies, doses, pregnancy/infant medicine safety and diagnosis "
                "requests get a fixed safety reply.\n"
                "3. Questions with nothing similar in the health dataset are declined.\n"
                "4. A near-identical question on file is answered with the clinician's answer.\n"
                "5. Otherwise the fine-tuned FLAN-T5 model writes the answer."
            )
        if not retriever.ready:
            st.warning("Retrieval index not loaded (no data/train.csv). Run `python scripts/data_prep.py`.")
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.session_state.last_question = None
            st.rerun()


if __name__ == "__main__":
    main()
