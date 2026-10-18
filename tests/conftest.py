import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))


@pytest.fixture(scope="session")
def retriever():
    from retrieval import Retriever
    return Retriever()


@pytest.fixture(scope="session")
def speller(retriever):
    from transformers import AutoTokenizer
    from query_cleanup import SpellCorrector
    tokenizer = AutoTokenizer.from_pretrained(str(REPO_ROOT / "model" / "flan-t5-base-afrimedqa")) \
        if (REPO_ROOT / "model" / "flan-t5-base-afrimedqa").exists() \
        else AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    return SpellCorrector(retriever.questions + retriever.answers,
                          known_words=SpellCorrector.words_from_tokenizer(tokenizer))
