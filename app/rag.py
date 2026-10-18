"""
Builds the input for retrieval-augmented generation (RAG; Lewis et al., 2020):
the user's question plus the top-k retrieved clinician answers as context,
fed to the fine-tuned FLAN-T5 (Experiment E). Used by scripts/evaluate_rag.py and
the notebook's Step 14. The app does NOT use it: Experiment E found retrieved
context didn't improve FLAN-T5 on the test set, so the app's FLAN-T5 mode
answers without context, and the chat model builds its own context
(app/chat_model.py).

The prompt format and k are not hand-picked: scripts/evaluate_rag.py tries
each option in PROMPT_FORMATS with k = 1, 3, 5 on the validation set and the
best one is written into DEFAULT_FORMAT / DEFAULT_K below. The test set is
only used afterwards, to report results.
"""

PREFIX = "answer the following health question in simple language for a general audience: "
MAX_PASSAGE_WORDS = 60   # long clinician answers would crowd out the others
MAX_RAG_INPUT_LEN = 512  # tokens; T5's usual input length, vs 256 without context

PROMPT_FORMATS = {
    # the fine-tuning prompt, unchanged, with context appended after it
    "question_then_context": lambda q, ctx: f"{PREFIX}{q} context: {ctx}",
    # FLAN-style reading-comprehension prompt (Chung et al., 2022)
    "context_then_question": lambda q, ctx: (
        f"Read the reference answers and use them to answer the health question in simple "
        f"language for a general audience.\n\nReference answers: {ctx}\n\nQuestion: {q}"
    ),
}
# chosen on the validation set by scripts/evaluate_rag.py (data/rag_val_tuning.csv):
# token F1 0.397 vs 0.362 for k=3 and 0.380 for the other format at k=1
DEFAULT_FORMAT = "context_then_question"
DEFAULT_K = 1


def _shorten(text: str) -> str:
    words = str(text).split()
    return " ".join(words[:MAX_PASSAGE_WORDS]) + (" ..." if len(words) > MAX_PASSAGE_WORDS else "")


def build_input(question: str, hits, fmt: str = DEFAULT_FORMAT) -> str:
    """hits: output of Retriever.search(). With no hits this falls back to
    the plain fine-tuning prompt."""
    question = question.replace(PREFIX, "").strip()
    if not hits:
        return PREFIX + question
    context = " | ".join(_shorten(h["answer"]) for h in hits)
    return PROMPT_FORMATS[fmt](question, context)
