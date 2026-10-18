"""
Deploys the chatbot on Modal (https://modal.com) with a public URL.

    pip install modal
    modal setup                      # once: log in through the browser
    modal deploy deploy/modal_app.py

The container uses a T4 GPU, so answers take seconds. It shuts down after 10 minutes without
visitors (so it only uses credit while someone is chatting) and starts again when the link is
opened; the first answer after that takes about a minute while the model loads.
"""

import subprocess
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1", "transformers==5.20.1", "peft==0.21.2", "accelerate==1.15.0",
        "sentence-transformers==6.1.0", "streamlit==1.66.0", "pandas==3.0.6", "numpy==2.4.6",
        "sentencepiece==0.2.2",
    )
    # download the base chat model and the retrieval encoder once, when the image is built
    .run_commands("python -c \"from huggingface_hub import snapshot_download as d; "
                  "d('Qwen/Qwen2.5-1.5B-Instruct'); d('sentence-transformers/all-MiniLM-L6-v2')\"")
    .add_local_dir(PROJECT / "app", f"{REMOTE}/app")
    .add_local_dir(PROJECT / "model" / "qwen2.5-1.5b-afrimedqa-lora", f"{REMOTE}/model/qwen2.5-1.5b-afrimedqa-lora")
    .add_local_file(PROJECT / "data" / "train.csv", f"{REMOTE}/data/train.csv")
    .add_local_file(PROJECT / "data" / "val.csv", f"{REMOTE}/data/val.csv")
    .add_local_file(PROJECT / "data" / "test.csv", f"{REMOTE}/data/test.csv")
)

app = modal.App("south-sudan-health-qa", image=image)


@app.function(gpu="T4", scaledown_window=600, timeout=24 * 60 * 60)
@modal.concurrent(max_inputs=20)
@modal.web_server(8000, startup_timeout=600)
def web():
    subprocess.Popen(
        f"streamlit run {REMOTE}/app/streamlit_app.py --server.port 8000 --server.address 0.0.0.0 "
        "--server.headless true --server.fileWatcherType none --browser.gatherUsageStats false "
        "--server.enableCORS false --server.enableXsrfProtection false",
        shell=True, cwd=REMOTE,
    )
