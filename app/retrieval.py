"""
Semantic retrieval over the AfriMed-QA training (+ validation) questions.

Two uses:

1. Retrieval-augmented generation (Experiment E, see scripts/evaluate_rag.py
   and app/rag.py): the top-k most similar clinician-answered questions are
   passed to the fine-tuned FLAN-T5 as context, so the model generates its
   answer from real reference material instead of only from what it
   memorised from 917 training examples (Step 9 of the notebook: token F1
   of only ~0.06 without context).
2. Direct answer: when a question is a near-duplicate of one already answered
   by a clinician (similarity >= DIRECT_ANSWER_THRESHOLD), the app shows that
   real answer as-is.

Questions are embedded with all-MiniLM-L6-v2 (Reimers & Gurevych, 2019,
Sentence-BERT), a small sentence-embedding Transformer, and compared with
cosine similarity. This replaced an earlier TF-IDF version, which only matched
on shared words: "how do you catch typhoid" and "how is typhoid transmitted"
share almost no words but mean the same thing. Runs on CPU in milliseconds.

Fine-tuning FLAN-T5 is still the project's core ML component — retrieval only
supplies context; the fine-tuned model still writes every generated answer.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PREFIX = "answer the following health question in simple language for a general audience: "
DIRECT_ANSWER_THRESHOLD = 0.85  # near-duplicate question: show the clinician answer itself


class Retriever:
    def __init__(self, splits=("train.csv", "val.csv"), encoder=None):
        """splits: which data/ CSVs to index. The app indexes all three
        (streamlit_app.APP_SPLITS); evaluation scripts index train + val;
        tuning on the validation set must index train only, so a validation
        question can't retrieve its own answer."""
        self.questions = []
        self.answers = []
        self.embeddings = None
        self.encoder = encoder
        self._load(splits)

    def _load(self, splits):
        frames = [pd.read_csv(DATA_DIR / s) for s in splits if (DATA_DIR / s).exists()]
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        # input_text carries the fixed instruction prefix; strip it so
        # similarity is computed on the question itself.
        self.questions = df["input_text"].astype(str).str.replace(PREFIX, "", regex=False).tolist()
        self.answers = df["target_answer"].astype(str).tolist()
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.embeddings = self._encode(self.questions)

    def _encode(self, texts):
        # normalised vectors, so a dot product is the cosine similarity
        return self.encoder.encode(texts, normalize_embeddings=True, batch_size=64)

    @property
    def ready(self) -> bool:
        return self.embeddings is not None and len(self.embeddings) > 0

    def search(self, question: str, k: int = 3):
        """Top-k most similar indexed questions, as a list of dicts with
        keys question, answer, similarity (highest first)."""
        if not self.ready:
            return []
        sims = self.embeddings @ self._encode([question.replace(PREFIX, "")])[0]
        top = np.argsort(-sims)[:k]
        return [
            {"question": self.questions[i], "answer": self.answers[i], "similarity": float(sims[i])}
            for i in top
        ]

    def retrieve(self, question: str):
        """Returns (answer, similarity, matched_question) for the closest
        indexed question if it is a near-duplicate, else (None, best_sim, None)."""
        hits = self.search(question, k=1)
        if not hits:
            return None, 0.0, None
        best = hits[0]
        if best["similarity"] >= DIRECT_ANSWER_THRESHOLD:
            return best["answer"], best["similarity"], best["question"]
        return None, best["similarity"], None
