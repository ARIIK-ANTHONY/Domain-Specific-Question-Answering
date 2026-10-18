"""Spelling correction, follow-ups and greetings (app/query_cleanup.py)."""

import pytest

from query_cleanup import resolve_followup, small_talk_reply


@pytest.mark.parametrize("typo, fixed", [
    ("systoms", "symptoms"), ("malria", "malaria"), ("maleria", "malaria"), ("diabetis", "diabetes"),
    ("typhiod", "typhoid"), ("sympthoms", "symptoms"), ("hedache", "headache"), ("pnemonia", "pneumonia"),
    ("vomitting", "vomiting"), ("cholra", "cholera"), ("tuberclosis", "tuberculosis"),
    ("stomak", "stomach"),      # was "stroma" with plain letter matching
    ("pregnent", "pregnant"),   # was "prevent", which hid pregnancy questions from the guardrail
])
def test_misspellings(speller, typo, fixed):
    assert speller.correct(typo) == fixed


@pytest.mark.parametrize("sentence", ["i mean it", "my body is hotting", "kala azar in jonglei", "pls help my son"])
def test_normal_words_left_alone(speller, sentence):
    assert speller.correct(sentence) == sentence   # "mean" once became "men"


def test_followups():
    assert resolve_followup("is that all", "what are the symptoms of malaria") == \
        ("what are the symptoms of malaria", "more")
    assert resolve_followup("how is it treated?", "what causes typhoid")[1] == "referring"
    assert resolve_followup("tell me more", None)[1] == "no_context"
    assert resolve_followup("what causes diabetes", "anything") == ("what causes diabetes", None)


@pytest.mark.parametrize("message", ["hey", "Hello there!", "good morning doctor", "thanks", "who are you"])
def test_greetings(message):
    assert small_talk_reply(message)


@pytest.mark.parametrize("message", ["hi what is malaria", "hey i have malaria what should i do"])
def test_questions_starting_with_a_greeting_are_not_small_talk(message):
    assert small_talk_reply(message) is None
