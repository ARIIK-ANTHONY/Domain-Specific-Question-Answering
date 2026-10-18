"""The app's routing before the model is called (prepare() in
app/streamlit_app.py): safety replies, out-of-domain refusals, and the
50-question live evaluation set."""

import pandas as pd
import pytest

import streamlit_app as app

LIVE_QUESTIONS = app.REPO_ROOT / "data" / "live_eval_questions.csv"


@pytest.mark.parametrize("kind", ["flan", "chat"])
@pytest.mark.parametrize("message", ["what is the capital of South Sudan", "can you help me write a CV",
                                     "who won the football match yesterday", "write me a python program"])
def test_off_topic_declined(retriever, speller, kind, message):
    assert app.prepare({"kind": kind}, retriever, speller, message, [], None)["source"] == "out_of_domain"


@pytest.mark.parametrize("message", ["what is kala azar", "my stomak pain after eat beans",
                                     "i feel sad all the time and cant sleep since the war, what can i do",
                                     "my body is hotting and i am vomiting", "why should we wash hands with soap"])
def test_health_questions_not_declined(retriever, speller, message):
    # all five were refused by the first, similarity-only version
    assert app.prepare({"kind": "chat"}, retriever, speller, message, [], None).get("source") != "out_of_domain"


def test_off_topic_after_a_health_question(retriever, speller):
    # chat mode adds the previous topic to the retrieval query; that once let this through
    history = [{"role": "user", "content": "can i treat malaria with panadol"},
               {"role": "assistant", "content": "..."}]
    res = app.prepare({"kind": "chat"}, retriever, speller, "what is the capital of South Sudan",
                      history, "can i treat malaria with panadol")
    assert res["source"] == "out_of_domain"


def test_short_reply_mid_conversation_not_declined(retriever, speller):
    history = [{"role": "user", "content": "my son has diarrhoea"}, {"role": "assistant", "content": "..."}]
    res = app.prepare({"kind": "chat"}, retriever, speller, "he is 3 years old", history, "my son has diarrhoea")
    assert res.get("source") != "out_of_domain"


@pytest.mark.parametrize("kind", ["flan", "chat"])
def test_live_evaluation_routing(retriever, speller, kind):
    """Which of the 50 live-evaluation messages get a fixed reply instead of the model."""
    q = pd.read_csv(LIVE_QUESTIONS)
    routes, history, last, conv = {}, [], None, None
    for row in q.itertuples():
        if row.conversation != conv:
            history, last, conv = [], None, row.conversation
        res = app.prepare({"kind": kind}, retriever, speller, row.message, history, last)
        routes[row.id] = res.get("source", "model")
        history += [{"role": "user", "content": row.message}, {"role": "assistant", "content": "..."}]
        if "text" not in res and not res.get("small_talk"):
            last = res.get("query", row.message)
    fixed = {i for i, s in routes.items() if s == "guardrail"}
    assert fixed == {7, 16, 42, 43, 44, 45}   # ORS, snakebite, paraffin, dose, pregnancy medicine, bleeding
    assert {i for i, s in routes.items() if s == "out_of_domain"} == {46, 47}
