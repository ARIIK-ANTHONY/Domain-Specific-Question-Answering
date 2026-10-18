"""The fixed safety replies. Several cases are regressions for problems found
while testing the app (see the comments in app/safety_guardrails.py)."""

import pytest

import safety_guardrails as g


def kind(message):
    reply = g.check(message)
    if reply is None:
        return None
    if reply == g.SELF_HARM_RESPONSE:
        return "self_harm"
    if reply == g.ORS_RESPONSE:
        return "ors"
    for name, marker in (("emergency", "medical emergency"), ("dose", "specific dose"),
                         ("pregnancy", "pregnancy, breastfeeding"), ("diagnosis", "diagnose you")):
        if marker in reply:
            return name
    return "other"


@pytest.mark.parametrize("message", [
    "I have chest pain", "my child is having a seizure", "she is unconscious",
    "a snake bit my brother on the leg what should we do", "my child swallowed paraffin what do i do",
    "my wife is bleeding a lot after giving birth", "bleeding heavily after delivery",
])
def test_emergencies(message):
    assert kind(message) == "emergency"


@pytest.mark.parametrize("message", [
    "I want to kill myself", "i feel suicidal", "i dont want to live anymore",
    "sometimes i think of ending my life", "i want to die",
])
def test_self_harm_gets_supportive_reply(message):
    assert kind(message) == "self_harm"


@pytest.mark.parametrize("message", ["can malaria kill", "will this disease kill me", "how do i kill mosquitoes"])
def test_self_harm_not_triggered_by_ordinary_questions(message):
    assert kind(message) != "self_harm"


def test_first_aid_tips():
    assert "Do not cut the bite" in g.check("a snake bit me")
    assert "Do not make the person vomit" in g.check("my child swallowed paraffin")


@pytest.mark.parametrize("message", ["what dose of paracetamol should I give my 2-year-old?",
                                     "how many tablets of coartem should i take",
                                     "can you write me a prescription for antibiotics"])
def test_doses_and_prescriptions(message):
    assert kind(message) == "dose"


@pytest.mark.parametrize("message", ["is it safe to take malaria medicine when pregnant",
                                     "is it safe to take ibuprofen in the third trimester",
                                     "can i give my 2 year old aspirin"])
def test_medicine_in_pregnancy_and_children(message):
    assert kind(message) == "pregnancy"


@pytest.mark.parametrize("message", ["how do i make ORS",
                                     "what is ORS and how do i make it at home if there is no pharmacy"])
def test_ors_recipe_is_fixed_who_recipe(message):
    assert kind(message) == "ors"
    assert "half a level teaspoon of salt" in g.ORS_RESPONSE


def test_diagnosis_only_blocked_when_requested():
    q = "i have flu, headache what sickness am i likely to have"
    assert kind(q) == "diagnosis"
    assert g.check(q, include_diagnosis=False) is None


@pytest.mark.parametrize("message", [
    "what are the symptoms of malaria", "how is typhoid diagnosed", "what is a snake",
    "is bleeding normal in the first trimester", "how do i prepare for pregnancy", "what is ORS",
])
def test_ordinary_questions_pass_through(message):
    assert kind(message) is None


def test_notes():
    assert g.fever_note("my body is hotting and i am vomiting") == g.FEVER_NOTE
    assert g.fever_note("my 8 month old baby has a fever") == g.FEVER_NOTE
    assert g.fever_note("what causes diabetes") is None
    assert g.medication_note("headache", "take ibuprofen") == g.MEDICATION_NOTE
    assert g.medication_note("what causes diabetes", "genes and lifestyle") is None
