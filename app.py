import os
import time
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "outputs/merged_model"
# Keep the data-shape the model expects, but remove the UI control for it
_FAVORITES_FIXED = 10

st.set_page_config(page_title="Elon Tweet Generator (Parody)", page_icon="🐦", layout="centered")

# -----------------------------
# Model Loading (cached)
# -----------------------------

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_path: str):
    """Load 4-bit quantized model and tokenizer, cached across reruns."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).eval()

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok

model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

# -----------------------------
# Session state (history)
# -----------------------------
if "tweets" not in st.session_state:
    st.session_state.tweets = []  # list of dicts: {"text": str, "ts": float}

# -----------------------------
# Helpers
# -----------------------------
def build_prompt(init_text) -> str:
    """
    Data expects:
        favorites: <N>\n
        <init_text> (optional)
    We keep favorites fixed under the hood; UI omits it per request.
    """
    prompt = f"favorites: {random.randint(1,50)}\n"
    if init_text:
        prompt += init_text.strip() + " "
    return prompt

@torch.inference_mode()
def generate_completion(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.99,
    top_p: float = 0.95,
    top_k: int = 60,
    min_new_tokens: int = 8
):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Slice off the prompt; show only completion
    input_len = inputs["input_ids"].shape[1]
    new_tokens = gen_out[0, input_len:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return completion.strip()

import time, html, textwrap

def tweet_card(text: str, ts: float):
    # 1) Escape and convert newlines
    safe_text = html.escape(text).replace("\n", "<br>")

    # 2) Dedent CSS/HTML so there are no leading spaces
    css = textwrap.dedent("""
    <style>
      .tweet-card { border:1px solid #e6ecf0; border-radius:12px; padding:16px; margin-bottom:14px; background:#fff; }
      .tweet-header { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
      .tweet-header img { width:48px; height:48px; border-radius:50%; }
      .tweet-names { color:black; line-height:1.1; }
      .tweet-name { color:black; font-weight:700; font-size:16px; }
      .tweet-handle { color:#536471; font-size:14px; }
      .tweet-body { color:black; font-size:18px; line-height:1.4; margin:6px 0 10px 0; }
      .tweet-meta { color:#536471; font-size:14px; }
    </style>
    """).strip()

    avatar_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZLZ0rWFYXARl_ZDTTJFhYivZXnfxecpEPqSxkzy-p3UwzWC6MyycxUctF&s"
    when = time.strftime("%b %d, %Y · %I:%M %p", time.localtime(ts))

    html_block = textwrap.dedent(f"""
    <div class="tweet-card">
      <div class="tweet-header">
        <img src="{avatar_url}" alt="Elon Musk">
        <div class="tweet-names">
          <div class="tweet-name">Elon Musk ✅</div>
          <div class="tweet-handle">@elonmusk</div>
        </div>
      </div>
      <div class="tweet-body">{safe_text}</div>
      <div class="tweet-meta">{when} · X</div>
    </div>
    """).strip()

    # 3) Use components.html (most robust), or st.markdown(..., unsafe_allow_html=True)
    # st.components.v1.html(css + html_block, height=200, scrolling=False)
    # Alternatively:
    st.markdown(css + html_block, unsafe_allow_html=True)
    
# -----------------------------
# UI
# -----------------------------
st.title("🐦 Elon Tweet Generator (Parody)")
st.caption("Parody app: generates tweets in a similar format to your training data. Not affiliated with the real Elon Musk.")

mode = st.radio(
    "Mode",
    ["Original tweet", "Finish my tweet"],
    horizontal=True,
)

# Temperature slider (front-and-center)
temperature = st.slider("Spice it up! 🔥🔥🔥 (Temperature)", 0.1, 3.0, 0.99, 0.01)

# Advanced settings remain optional
with st.expander("Advanced generation settings"):
    max_new_tokens = st.slider("Max new tokens", 8, 512, 128, 8)
    top_p          = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.95, 0.01)
    top_k          = st.slider("Top-k", 0, 200, 60, 5)
    min_new_tokens = st.slider("Min new tokens", 0, 64, 32, 1)

init_text = ""
tweet_list = ['I am known ',
              'I love ',
              'People always want ',
              'Does anybody ',
              'Sam Altman ', 
              'Donald Trump ',
             ]
if mode == "Finish my tweet":
    init_text = st.text_area(
        "Half-written tweet to finish",
        value= 'My tweet', #random.choice(tweet_list),
        help="Provide a partial tweet. The model will continue it.",
        height=120,
    )

col1, col2 = st.columns([1, 1])
with col1:
    go = st.button("Generate tweet", type="primary")
with col2:
    if st.button("Clear history"):
        st.session_state.tweets.clear()
        st.success("History cleared.")

# -----------------------------
# Action
# -----------------------------
if go:
    with st.spinner("Generating…"):
        prompt = build_prompt(init_text=None if mode == "Original tweet (no init)" else init_text)
        completion = generate_completion(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_new_tokens=min_new_tokens,
        )
        final_tweet = (init_text.strip() + " " if init_text else "") + completion
        st.session_state.tweets.insert(0, {"text": final_tweet, "ts": time.time()})

st.markdown("---")
st.subheader("Timeline")

if not st.session_state.tweets:
    st.info("No tweets yet. Generate one!")
else:
    for t in st.session_state.tweets:
        tweet_card(t["text"], t["ts"])

st.caption("Tip: If completions stop too early, increase `Min new tokens` or temperature. If they ramble, reduce temperature/top-p.")