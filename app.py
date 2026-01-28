import streamlit as st
import torch
from diffusion_wrapper import MedicalDiffusionModel
from transformers import AutoTokenizer
import numpy as np
import time

# --- Page Config ---
st.set_page_config(
    page_title="Medical Diffusion AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
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

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 30522
    d_model = 512
    seq_len = 128
    
    model = MedicalDiffusionModel(vocab_size=vocab_size, d_model=d_model, max_seq_len=seq_len).to(device)
    
    # Try to load weights if they exist
    try:
        model.load_state_dict(torch.load("medical_diffusion.pt", map_location=device))
        st.sidebar.success("✅ Model weights loaded from medical_diffusion.pt")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Weights file not found. Running with randomly initialized weights.")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer, device

def generate_note(model, tokenizer, device, diagnosis, steps=50):
    model.eval()
    with torch.no_grad():
        # Encode condition
        diag_enc = tokenizer(
            diagnosis,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        cond_emb = model.backbone.embedding(diag_enc["input_ids"]).mean(dim=1)
        
        # Temporary set steps for faster demo if requested, 
        # but the generate method uses self.diffusion_pipeline.num_steps which is 1000 by default.
        # For the demo, let's keep it default or allow user to tweak if we modified the class.
        # Since we can't easily modify the class instance variables safely across sessions without cache issues,
        # we'll use the default 1000 steps but show a progress bar.
        
        with st.status("🧬 Generating medical note...", expanded=True) as status:
            st.write("Initializing reverse diffusion...")
            # We can't easily inject 'steps' into the existing model.generate without modifying it,
            # but we can provide a progress bar by manually iterating if we wanted.
            # For now, let's use the model's generate method.
            
            start_time = time.time()
            gen_embs = model.generate(cond_emb, seq_len=128, device=device)
            tokens = model.decode_embeddings(gen_embs)
            decoded_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            
            end_time = time.time()
            status.update(label=f"Done in {end_time - start_time:.2f}s!", state="complete", expanded=False)
            
    return decoded_text

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
steps_slider = st.sidebar.slider("Diffusion Steps (Visual only)", 10, 1000, 1000, help="The model currently uses 1000 steps by default.")

# --- Main UI ---
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

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: #888;'>Powered by Shadow Gravity Diffusion Engine</div>", 
    unsafe_allow_html=True
)
