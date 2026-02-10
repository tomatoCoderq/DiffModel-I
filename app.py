import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from diffusion_wrapper import MedicalDiffusionModel
from training_ipc import (
    enqueue_start,
    enqueue_stop,
    get_metrics,
    get_status,
    init_db,
    is_worker_running,
    reset_metrics,
    start_worker_if_needed,
)

st.set_page_config(
    page_title="Medical Diffusion AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4facfe;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Мы перезагружаем AutoTokenizer, если импорт transformers оказался частично выполнен
def _safe_load_autotokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError:
        import importlib
        import sys
        time.sleep(0.2)
        sys.modules.pop("transformers", None)
        AutoTokenizer = importlib.import_module("transformers").AutoTokenizer
        return AutoTokenizer


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 30522
    d_model = 512
    seq_len = 128

    model = MedicalDiffusionModel(vocab_size=vocab_size, d_model=d_model, max_seq_len=seq_len).to(device)

    try:
        model.load_state_dict(torch.load("medical_diffusion.pt", map_location=device))
        st.sidebar.success("✅ Model weights loaded from medical_diffusion.pt")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Weights file not found. Running with randomly initialized weights.")

    AutoTokenizer = _safe_load_autotokenizer()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer, device

def generate_note(model, tokenizer, device, diagnosis, steps=50):
    model.eval()
    with torch.no_grad():
        diag_enc = tokenizer(
            diagnosis,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        cond_emb = model.backbone.embedding(diag_enc["input_ids"]).mean(dim=1)

        # Мы оставляем число шагов равным настройке пайплайна, потому что кеш Streamlit не пересоздает объект генерации

        with st.status("🧬 Generating medical note...", expanded=True) as status:
            st.write("Initializing reverse diffusion...")
            start_time = time.time()
            gen_embs = model.generate(cond_emb, seq_len=128, device=device)
            tokens = model.decode_embeddings(gen_embs)
            decoded_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

            end_time = time.time()
            status.update(label=f"Done in {end_time - start_time:.2f}s!", state="complete", expanded=False)

    return decoded_text

st.sidebar.title("⚙️ Settings")
steps_slider = st.sidebar.slider("Diffusion Steps (Visual only)", 10, 1000, 1000, help="The model currently uses 1000 steps by default.")

st.sidebar.divider()
st.sidebar.subheader("🧪 Training Control")

init_db()
worker_started_on_load = start_worker_if_needed()
worker_running = is_worker_running()
if worker_started_on_load:
    st.sidebar.info("Training worker started on app load.")
st.sidebar.write(f"Worker: {'running' if worker_running else 'not running'}")

data_path = st.sidebar.text_input("Dataset path", value="expanded_medical_dataset.csv")
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=512, value=32, step=1)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=10000, value=200, step=1)
learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-7, max_value=1e-2, value=5e-5, step=1e-5, format="%.7f")
status_every_steps = st.sidebar.number_input("Status update steps", min_value=1, max_value=1000, value=10, step=1)

col_start, col_stop = st.sidebar.columns(2)
with col_start:
    if st.button("Start Training"):
        worker_started = start_worker_if_needed()
        reset_metrics()
        enqueue_start(
            {
                "data_path": data_path,
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "status_every_steps": int(status_every_steps),
            }
        )
        if worker_started:
            st.sidebar.info("Training worker started.")
        st.sidebar.success("Training start requested.")

with col_stop:
    if st.button("Stop Training"):
        enqueue_stop()
        st.sidebar.warning("Stop requested.")

if st.sidebar.button("Refresh Status"):
    st.rerun()

status = get_status()
if status:
    st.sidebar.markdown("**Status**")
    st.sidebar.write(f"State: {status.get('state')}")
    st.sidebar.write(f"Detail: {status.get('detail')}")
    st.sidebar.write(f"Epoch: {status.get('epoch')}")
    st.sidebar.write(f"Step: {status.get('step')}")
    st.sidebar.write(f"Loss: {status.get('loss')}")
    st.sidebar.write(f"Updated: {status.get('updated_at')}")
else:
    st.sidebar.info("No training status found. Start worker: `python training_worker.py`.")

st.sidebar.divider()
st.sidebar.subheader("⏱️ Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable auto refresh", value=False)
refresh_seconds = st.sidebar.number_input("Refresh interval (sec)", min_value=1, max_value=60, value=5, step=1)
if auto_refresh:
    time.sleep(int(refresh_seconds))
    st.rerun()

st.title("🧬 Medical Diffusion AI")
st.markdown("### Conditional Clinical Note Generation")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="status-card">
        <h4>Input Diagnosis</h4>
        Enter a diagnosis or patient condition to generate a corresponding medical note.
        </div>
    """, unsafe_allow_html=True)

    diagnosis_input = st.text_area(
        "Diagnosis / Condition:",
        placeholder="e.g., Acute Coronary Syndrome, Hypertension, Post-Appendectomy...",
        height=150
    )

    generate_btn = st.button("Generate Note")

with col2:
    st.markdown("""
        <div class="status-card">
        <h4>Generated Clinical Note</h4>
        The result will appear here. Note: If weights are not trained, the output will be random tokens.
        </div>
    """, unsafe_allow_html=True)

    if generate_btn:
        if diagnosis_input.strip() == "":
            st.error("Please enter a diagnosis.")
        else:
            model, tokenizer, device = load_model()
            result = generate_note(model, tokenizer, device, diagnosis_input)

            st.success("Generation Complete!")
            st.markdown(f"**Result:**")
            st.code(result, language="text")

            with st.expander("Technical Metadata"):
                st.write(f"Device: {device}")
                st.write(f"Vocabulary: {tokenizer.vocab_size}")
                st.write(f"Sequence Length: 64")

st.markdown("### 📈 Training Loss")
metrics = get_metrics(limit=300)
if metrics:
    df = pd.DataFrame(metrics)
    df = df.dropna(subset=["step", "loss"]).sort_values("step")
    if not df.empty:
        df = df.set_index("step")
        st.line_chart(df["loss"])
else:
    st.info("No training metrics yet.")

st.divider()
st.markdown(
    "<div style='text-align: center; color: #888;'>Powered by Shadow Gravity Diffusion Engine</div>",
    unsafe_allow_html=True
)
