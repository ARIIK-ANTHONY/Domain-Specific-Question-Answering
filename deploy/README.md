---
title: South Sudan Health QA
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: cc-by-nc-sa-4.0
short_description: Health question answering for South Sudan (AfriMed-QA)
---

# Healthcare Question-Answering Assistant for South Sudan

A student capstone project (ALU NLP). Qwen2.5-1.5B-Instruct fine-tuned with LoRA on
AfriMed-QA v2 clinician answers, with retrieval of similar clinician answers and safety rules
for emergencies, medicine doses and self-harm.

**Not a medical tool.** It gives general health information only and can be wrong.

This free Space runs on a CPU, so each answer takes about 30-90 seconds.
