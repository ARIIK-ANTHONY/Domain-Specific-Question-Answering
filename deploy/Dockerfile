# Hugging Face Space (Docker SDK, free CPU: 2 vCPU, 16 GB RAM) for the South Sudan health QA chatbot.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch (much smaller than the CUDA build), then the app's libraries
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Download the base chat model and the retrieval encoder at build time, so the app starts faster
RUN python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct'); d('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
