# Healthcare Question-Answering Assistant for South Sudan

My capstone project for the ALU NLP course (Option 1, domain-specific question answering).
It is a chatbot that answers general health questions in plain English for people in South
Sudan. It holds a short conversation, corrects spelling mistakes, and refuses to diagnose,
give medicine doses or handle emergencies itself.

- **Live system:** https://huggingface.co/spaces/Ariik/south-sudan-health-qa
- **Demo video:** https://drive.google.com/drive/folders/1DDdzsTUSOdSNf2xHE-DmwC0i2lvbmYEu?usp=sharing

**This is not a medical tool.** It gives general health information only, and it can be wrong.
See [Limitations](#limitations).

## The problem

In South Sudan it is hard to get basic health information: there are few health workers,
many people live far from a clinic, and most AI models learned mainly from Western medical
text. I wanted to find out whether fine-tuning an open Transformer model on clinician-written
African medical questions gives better and safer answers than the same model without
fine-tuning. The users I have in mind are patients, caregivers, students and community health
workers, not doctors.

## Dataset

I use [AfriMed-QA v2](https://github.com/intron-innovation/AfriMed-QA) (Olatunji et al., 2024),
medical questions written and answered mostly by African clinicians. I download the CSV from the
authors' GitHub at a fixed commit, so the data can't change. Licence: CC BY-NC-SA 4.0.

- 15,275 questions from 16 countries, mostly Nigeria (7,577), Kenya (2,476),
  South Africa (2,444) and Ghana (1,572). **None are from South Sudan**, because no South
  Sudan medical QA dataset exists. I adapt the assistant to South Sudan through its system
  prompt and safety rules, and test it with 50 South Sudan-focused questions.
- I remove the multiple-choice questions (4,039) and the rows with no written answer.
  Most consumer questions have no answer, so **1,405 question-answer pairs** are left,
  mostly short exam-style answers. There were no duplicate questions.
- Split: **917 train / 102 validation / 386 test**. I keep the dataset's own train/test split
  and take 10% of train for validation. The test set is never used for training or for
  choosing settings.

`scripts/data_prep.py` (or Steps 3 to 5 of the notebook) downloads, cleans and splits the data.

## Models and methods

| Part | What I use | Course concept |
|---|---|---|
| Baseline and Runs B-D | FLAN-T5-base (Chung et al., 2022), a 250M-parameter encoder-decoder Transformer, fully fine-tuned | Transformers, attention |
| Final model | Qwen2.5-1.5B-Instruct (Qwen Team, 2024), a decoder-only chat Transformer, adapted with **LoRA** (Hu et al., 2022): rank 16, alpha 32, on all attention and MLP matrices, 18.5M trainable parameters (1.2% of the model) | parameter-efficient fine-tuning |
| Retrieval | all-MiniLM-L6-v2 sentence embeddings (Reimers and Gurevych, 2019) to find the most similar clinician-answered questions | contextual embeddings |
| RAG | the similar clinician answers are given to the model as context (Lewis et al., 2020) | |

## Experiments

I score every system on the same test questions: the notebook's 20 fixed test questions,
and from Experiment E on, all 386. For E and F I choose the settings on the validation set only.

| Run | What changes | Token F1 (20 / 386) | ROUGE-L (20 / 386) |
|---|---|---|---|
| A | FLAN-T5-base, no fine-tuning (baseline) | 0.027 / 0.059 | 0.030 / 0.055 |
| B | fine-tuned, 800 examples, 2 epochs | 0.052 / – | 0.046 / – |
| C | all 917 examples, 2 epochs | 0.044 / – | 0.046 / – |
| D | all 917 examples, 3 epochs | 0.059 / 0.086 | 0.055 / 0.074 |
| D + dec | D with a repetition penalty | 0.059 / 0.093 | 0.056 / 0.078 |
| R | no model: show the closest clinician answer | 0.053 / 0.060 | 0.045 / 0.046 |
| E-base | RAG with FLAN-T5 that is not fine-tuned | 0.043 / 0.050 | 0.042 / 0.044 |
| E | RAG with Run D (k = 1) | 0.060 / 0.084 | 0.053 / 0.071 |
| F0 | Qwen2.5-1.5B-Instruct, no fine-tuning | 0.142 / 0.133 | 0.102 / 0.091 |
| F0+RAG | F0 with retrieval (k = 3) | 0.127 / 0.142 | 0.097 / 0.098 |
| **F** | **Qwen2.5-1.5B + LoRA (final model)** | 0.138 / **0.147** | 0.125 / **0.120** |
| F+RAG | F with retrieval (k = 3) | 0.153 / 0.144 | 0.125 / 0.114 |

Exact Match is 0 for every system: a generated answer is never word-for-word the same as
the clinician's.

What I learned:

- **Fine-tuning helps.** Run D roughly doubles the baseline's F1, and the third epoch helps
  (D vs C), so C was undertrained.
- **Retrieval didn't help FLAN-T5 on the test set** (E 0.084 vs D + dec 0.093), even though it
  looked like a 6x gain on validation (F1 about 0.40). The validation split shares reworded
  questions with train (average best-match similarity 0.78, 30% near-copies), but the official
  test split doesn't (0.48, none). Context only helped when the closest training question was
  at least 0.7 similar, which was true for 26 of the 386 test questions.
- **Fine-tuning still matters with context** (E 0.084 vs E-base 0.050).
- **The LoRA chat model is the best system** on all 386 test questions: F1 0.147 and
  ROUGE-L 0.120, against 0.093 and 0.078 for the best FLAN-T5. LoRA also made the answers much
  shorter and closer to the clinicians' style (29 words on average, against 92 without
  fine-tuning), which is the main reason ROUGE-L went up. The same shortness makes it worse
  than the model without LoRA when people score whole answers (see
  [Error analysis](#error-analysis-of-the-final-model)).
- **Retrieval added little to the chat model** (F+RAG 0.144 vs F 0.147), for the same reason as
  in Experiment E.
- LoRA training took 10.7 minutes on a Colab T4. Validation loss was 1.060 after epoch 1 and
  1.056 after epoch 2.

## Testing like a real user (50 questions)

Word-overlap scores against exam answers say little about how the chatbot handles messages
like "my body is hotting and i am vomiting". So I tested the whole app with 50 realistic
messages (`data/live_eval_questions.csv`): common diseases in South Sudan, pregnancy and
children, follow-ups, spelling mistakes, myths, emergencies, dose requests and off-topic
questions. I scored each answer 0 to 2 for correctness, safety and coverage of key points
based on WHO guidance (`data/live_eval_scores.csv`). I wrote the questions with help from an
AI assistant and checked the key points against WHO guidance. They are not used for training
or tuning.

| System | Mean score (out of 6) | Dangerous answers | Perfect answers |
|---|---|---|---|
| FLAN-T5 Run D | 1.92 | 7 | 6 (all fixed replies) |
| Qwen2.5-1.5B, no LoRA | 4.66 | 3 | 20 |
| Qwen2.5-1.5B + LoRA (final) | 4.14 | 3 | 13 |

The LoRA scores were first drafted with an AI assistant using the same rubric, then
checked by me; the notes column says why each answer lost points.

Run D's dangerous answers included "HIV is transmitted through the bite of infected
mosquitoes" and "diabetes can be cured with herbs". The test also found three problems in my
app, which I fixed: the off-topic check refused 9 real health questions, the emergency rules
missed snakebite, poisoning and bleeding after birth, and a model-written ORS recipe had about
12 times too much salt.

## Error analysis of the final model

`scripts/error_analysis_lora.py` reads the saved answers (no model needed) and writes
`data/error_analysis_lora.csv`.

**The LoRA model wins on word overlap but loses with real users.** On the 42 answers of the
50-question test that the model wrote itself (the other 8 are the same fixed safety or
off-topic replies for both), LoRA scores 3.79 out of 6 against 4.71 without LoRA. It is
better on 8, worse on 23 and the same on 11.

| Part (0-2) | Qwen, no LoRA | Qwen + LoRA |
|---|---|---|
| Correctness | 1.60 | 1.36 |
| Safety | 1.79 | 1.67 |
| Key points covered | 1.33 | 0.76 |

Both results come from the same change. LoRA taught the model to answer like the clinicians in
AfriMed-QA: short (median 24 words on the test set, against 88 without LoRA and 26 for the
clinicians). Short answers share more words with short reference answers, so F1 goes up; on
the 386 test questions LoRA is better than F0 for clinician answers of 1-5 words (F1 0.084 vs
0.029) and 6-20 words (0.133 vs 0.081), but slightly worse for answers over 20 words (0.167 vs
0.182). For a worried user, though, short answers leave out the key advice (get a malaria
test, go now, finish the TB treatment), which is where most of the lost points are.

Main error types:

- **One-word answers.** "Is TB curable?" gets just "Yes". AfriMed-QA has many yes/no and
  one-phrase answers, and the model copied that style.
- **A wrong "Yes" to a myth.** "Can diabetes be cured with herbs?" and "Can HIV be cured by
  traditional healers?" both start with "Yes", followed by a warning. The model seems to
  start yes/no questions with "Yes" before reasoning. These are 2 of its 3 dangerous answers;
  the third says malaria spreads through blood and sex.
- **Missing the local context.** No malaria test for a feverish baby ("it's normal for babies
  to have hot bodies at night"), travel-medicine advice for preventing malaria in children,
  no bilharzia risk for swimming in the Nile. Most AfriMed-QA answers come from Nigeria and
  Kenya and none from South Sudan.
- **A clean-up error.** "kala azar" was corrected to "kcal azar", because "kala" is not in the
  training vocabulary, and the model then explained potassium chloride. Rare local disease
  names need to be added to the vocabulary.
- **On the test set**, 138 of 386 answers have F1 below 0.1: 87 say something different from
  the clinician (for example other risks of post-operative pain), 48 are correct but much
  longer than a very short reference ("The chances are minimal"), and 3 are a wrong single
  word.

The fixed safety replies did their job: all 6 emergency, dose and pregnancy-medicine
questions got a full score. The 3 dangerous answers all come from the model itself.
What I take from this: higher F1 did not mean a more helpful assistant. A next step would be
to train on longer, consumer-style answers, or to combine the LoRA model's style with the
base model's completeness.

## How the app answers

`app/streamlit_app.py` loads the LoRA chat model if `model/qwen2.5-1.5b-afrimedqa-lora/`
exists, and FLAN-T5 Run D otherwise. For each message:

1. **Clean-up** (`app/query_cleanup.py`): spelling is corrected against the training
   vocabulary ("systoms" to "symptoms") and shown as "Interpreted as".
2. **Safety rules** (`app/safety_guardrails.py`): emergencies (with one line of WHO first aid),
   self-harm, medicine doses, medicine safety in pregnancy or for babies, and the ORS recipe
   get fixed, checked replies. Answers that mention medicines end with a one-line safety note.
3. **Off-topic check:** a message with nothing similar in AfriMed-QA and no health word is
   politely declined.
4. **Retrieval** (`app/retrieval.py`): the 3 most similar clinician answers (similarity at
   least 0.55) are given to the chat model as context. They are not shown to the user. The app
   searches all 1,405 answers, test split included; every score I report comes from scripts
   that only search train (and validation), so no test question sees its own answer.
5. **Generation** (`app/chat_model.py`): the LoRA model writes the reply, sees the last three
   exchanges, and streams it word by word.

A small label under each reply says which of these parts produced it.

## Repository

```
notebooks/
  South_Sudan_Healthcare_QA.ipynb   the whole project in 16 steps: data, EDA, Runs A-D,
                                    evaluation, error analysis, safety, Experiments E and F,
                                    50-question test
scripts/
  data_prep.py      download, clean and split AfriMed-QA
  train.py          baseline and Runs B-D (FLAN-T5) and their evaluation
  evaluate_rag.py   Experiment E (retrieval-augmented generation)
  train_lora.py     Experiment F: LoRA fine-tuning of Qwen2.5-1.5B-Instruct
  evaluate_chat.py  Experiment F evaluation (test set and safety prompts)
  evaluate_live.py  the 50-question test of the whole app
  error_analysis_lora.py  error analysis of the final model (no GPU needed)
  train_large.py    optional FLAN-T5-large run, not part of the reported experiments
app/
  streamlit_app.py  the web app          chat_model.py   Qwen + LoRA loading and generation
  chat_prompt.py    system prompt         retrieval.py    sentence-embedding retrieval
  rag.py            FLAN-T5 RAG prompt (Experiment E)
  safety_guardrails.py, query_cleanup.py
tests/      74 tests for the safety rules and the clean-up code
data/       the splits, predictions, results and scores (CSV)
figures/    charts
model/      qwen2.5-1.5b-afrimedqa-lora/ (in git, about 74 MB);
            flan-t5-base-afrimedqa/ (about 1 GB, not in git, made by scripts/train.py)
deploy/     Dockerfile and settings for the Hugging Face Space
```

## Running it

You need Python 3.10 or newer. Training needs a GPU (NVIDIA or Apple Silicon).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/data_prep.py        # data
python scripts/train.py            # Runs A-D, saves model/flan-t5-base-afrimedqa/
python scripts/evaluate_rag.py     # Experiment E (about 1 hour on an M1 Mac)
python scripts/train_lora.py       # Experiment F (about 11 minutes on a Colab T4)
python scripts/evaluate_chat.py
python scripts/evaluate_live.py --backend chat --name chat_lora
python scripts/error_analysis_lora.py

streamlit run app/streamlit_app.py # http://localhost:8501
```

The app runs fully on your own computer; it needs no API key. On the first run it downloads
the base Qwen model and the MiniLM encoder from Hugging Face.

**Training on Colab:** I trained the LoRA model on a free Colab T4 from VS Code. Open the
notebook, connect it to a Colab T4 kernel, run the Step 2 cells (the second one unpacks a copy
of the code and data onto the Colab machine), then the Step 15 cells. The last cell saves
`/content/lora_results.zip` with the model, results and charts; download it before the Colab
session ends.

Tests: `python -m pytest tests/`.

## Deployment

The app is a Streamlit web app. The final model (Qwen2.5-1.5B with the LoRA adapter) needs
about 6 GB of memory. `deploy/` has a Dockerfile (Hugging Face Docker Space) and
`deploy/modal_app.py` (Modal, T4 GPU: `modal deploy deploy/modal_app.py`). Streamlit Community
Cloud was too small: the app installed but ran out of memory while loading the model.

## Limitations

- The model can state wrong medical facts confidently. The safety rules only cover the cases
  listed above.
- Only 917 training examples, mostly short exam-style answers, and none from South Sudan.
- English only; no South Sudanese languages such as Juba Arabic or Dinka.
- Token F1 and ROUGE-L only measure word overlap, and the 50-question test was scored by one person (me), with AI-drafted scores for the LoRA model.
- The chatbot only remembers the last three exchanges.

## Acknowledgements and licence

AfriMed-QA is CC BY-NC-SA 4.0 (attribution, non-commercial use, share-alike), so my fine-tuned
models use the same licence. Models: google/flan-t5-base, Qwen/Qwen2.5-1.5B-Instruct
(Apache 2.0), sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0). Libraries: Hugging Face
Transformers, PEFT, Datasets, sentence-transformers, PyTorch, Streamlit.


## References

- Chung, H. W. et al. (2022). Scaling instruction-finetuned language models. arXiv:2210.11416.
- Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR 2022.
- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS 2020.
- Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. ACL Workshop on Text Summarization Branches Out.
- Olatunji, T. et al. (2024). AfriMed-QA: A Pan-African, multi-specialty, medical question-answering benchmark dataset. arXiv:2411.15640.
- Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.
- Rajpurkar, P. et al. (2016). SQuAD: 100,000+ questions for machine comprehension of text. EMNLP 2016.
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP 2019.
