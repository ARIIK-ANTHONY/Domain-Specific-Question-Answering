"""
Light pre-processing of the user's question, added after live testing of the
app showed two failure modes that the notebook's evaluation never exercised
(its test questions were all clean, standalone questions):

    - typos: "what are the systoms of malaria" got a confused answer from the
      fine-tuned model and no retrieval match, while the correctly spelled
      question got a sensible answer. "systoms" never appears in training, so
      both the retriever and the model treat it as noise.
    - follow-ups: "is that all" after a malaria answer produced an unrelated
      answer about iodine deficiency. The model is single-turn — it only ever
      sees the current message — so a follow-up with no topic in it gives it
      nothing to answer. resolve_followup() folds the previous question in.

Spelling correction uses only the vocabulary of the AfriMed-QA training and
validation questions/answers (difflib, standard library), so it can only
"correct" towards words the model was actually trained on. Follow-ups are
detected with a short, auditable phrase list; with no earlier question to
refer back to, they get a fixed message asking for the full question,
rather than letting the model guess.
"""

import difflib
import re
from collections import Counter

MIN_WORD_COUNT = 2   # ignore one-off words in the data (often typos themselves)
MIN_WORD_LEN = 4     # don't touch short words like "is", "flu", "hiv"
CUTOFF = 0.8         # similarity of sound-alike keys needed to accept a correction
TIE_MARGIN = 0.06    # among near-equal candidates, prefer the more common word
                     # ("systoms" is slightly closer to "systems" than "symptoms")
# Similarity is computed on rough sound-alike keys, because people misspell by
# sound: plain letter matching turned "stomak" into "stroma" and "pregnent" into
# "prevent" (which would have hidden a pregnancy question from the guardrail).
SOUND_ALIKE = (("ch", "k"), ("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"))


def _sound_key(word: str) -> str:
    word = word.lower()
    for written, sound in SOUND_ALIKE:
        word = word.replace(written, sound)
    return re.sub(r"(.)\1+", r"\1", word)  # "vomitting" -> "vomiting"

FOLLOWUP_PATTERNS = [
    r"^is that all\b", r"^anything else\b", r"^what else\b", r"^(tell me )?more\b",
    r"^(can you )?(explain|elaborate)( more| further)?\W*$", r"^and\W*$",
    r"^(what about|how about) (it|that|this|them)\b", r"^why\W*$", r"^really\W*$",
    r"^go on\b", r"^continue\b", r"^(and )?then\W*$", r"^is there more\b",
    r"^any other\W*$", r"^others?\W*$",
]

FOLLOWUP_RESPONSE = (
    "I'm not sure what you're referring to. Please ask the full question, "
    "for example \"What are the symptoms of malaria?\" or \"How is malaria treated?\"."
)


# short questions that point back at the previous topic: "how is it treated?"
REFERRING_WORDS = r"\b(it|its|this|that|these|those|they|them|their|the disease|the condition|the illness)\b"
MAX_REFERRING_WORDS = 8


# Greetings and small talk. Live testing: "hey" was answered with the
# out-of-domain message, which feels broken in a chat interface.
SMALL_TALK = [
    (r"^(hi|hey|hello|hallo|hiya|good (morning|afternoon|evening)|salam|salaam)"
     r"( (there|doctor|doc|everyone|sir|madam|friend))?\W*$",
     "Hello! I can answer general health questions, for example about malaria, typhoid, "
     "cholera, diabetes, high blood pressure, pregnancy care or nutrition. What would you "
     "like to know?"),
    (r"^how are you\b|^how('?s| is) it going\b",
     "I'm doing well, thank you for asking. Is there a health question I can help you with?"),
    (r"^(thanks|thank you|thx|ok thanks|okay thank you)\b",
     "You're welcome. If anything gets worse or you're worried, please see a healthcare worker."),
    (r"^(bye|goodbye|see you)\b",
     "Goodbye, and take care. Remember to visit a health facility if your symptoms get worse."),
    (r"^(who are you|what are you|what can you do|what do you do|help)\W*$",
     "I'm a health information assistant trained on AfriMed-QA, a collection of health "
     "questions answered by clinicians across Africa. I can explain illnesses, symptoms, "
     "causes, prevention and treatment in general terms. I can't diagnose you, give "
     "medicine doses or handle emergencies, so for those please see a healthcare worker."),
]


def small_talk_reply(message: str):
    """Fixed reply for greetings / thanks / "who are you", else None."""
    m = message.lower().strip()
    for pattern, reply in SMALL_TALK:
        if re.search(pattern, m):
            return reply
    return None


def is_followup(question: str) -> bool:
    q = question.lower().strip()
    return any(re.search(p, q) for p in FOLLOWUP_PATTERNS)


def resolve_followup(question: str, previous_question):
    """Lightweight conversation memory. The model itself is single-turn, so
    the previous question is folded into the current one before retrieval
    and generation. Returns (question_to_answer, kind) where kind is:
      None         - a standalone question, unchanged
      "more"       - "is that all" / "tell me more": re-answer the previous
                     question with more retrieved context
      "referring"  - "how is it treated?": previous question added as topic
      "no_context" - a follow-up with nothing to follow (first message)
    """
    if is_followup(question):
        return (previous_question, "more") if previous_question else (question, "no_context")
    q = question.lower().strip()
    if previous_question and len(q.split()) <= MAX_REFERRING_WORDS and re.search(REFERRING_WORDS, q):
        return f"{question.strip().rstrip('?')} ({previous_question.strip().rstrip('?')})", "referring"
    return question, None


class SpellCorrector:
    def __init__(self, texts, known_words=()):
        """texts: training questions/answers, the only words we correct
        towards. known_words: ordinary English words (the model tokenizer's
        whole-word vocabulary) that are never "corrected" even if absent from
        the medical data — otherwise "i mean" became "i men"."""
        counts = Counter(w for t in texts for w in re.findall(r"[a-z]+", t.lower()))
        self.counts = counts
        self.known = set(known_words)
        self.vocab = [w for w, n in counts.items() if n >= MIN_WORD_COUNT and len(w) >= 3]

    def _fix_word(self, word: str) -> str:
        lower = word.lower()
        if len(lower) < MIN_WORD_LEN or lower in self.counts or lower in self.known:
            return word
        key = _sound_key(lower)
        scored = [
            (difflib.SequenceMatcher(None, key, _sound_key(cand)).ratio(), cand)
            for cand in difflib.get_close_matches(lower, self.vocab, n=10, cutoff=0.6)
        ]
        scored = [(r, c) for r, c in scored if r >= CUTOFF]
        if not scored:
            return word
        best_ratio = max(r for r, _ in scored)
        near_best = [c for r, c in scored if r >= best_ratio - TIE_MARGIN]
        return max(near_best, key=lambda c: self.counts[c])

    @staticmethod
    def words_from_tokenizer(tokenizer):
        """Whole English words in a tokenizer's vocabulary. SentencePiece
        (FLAN-T5) marks a word start with "\u2581" ("\u2581mean"), byte-level BPE
        (Qwen) with "\u0120" ("\u0120mean")."""
        words = set()
        for piece in tokenizer.get_vocab():
            word = piece.lstrip("\u2581\u0120")
            if word != piece and word.isalpha():
                words.add(word.lower())
        return words

    def correct(self, question: str) -> str:
        return re.sub(r"[A-Za-z]+", lambda m: self._fix_word(m.group(0)), question)
