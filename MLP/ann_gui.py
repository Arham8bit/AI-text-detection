"""
ann_gui.py  —  Streamlit GUI for BERT-MLP AI Text Detector
Run: streamlit run ann_gui.py

Requires: streamlit, torch, transformers, numpy, pickle, joblib, matplotlib, scikit-learn
Models:   bert_pca.pkl, mlp_scaler.pkl, mlp_model.pkl, mlp_label_encoder.pkl
"""

import json
import pickle
import warnings
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANN · AI Text Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0c0f14;
    color: #dde1e7;
}

/* Header */
.ann-header {
    background: linear-gradient(135deg, #130f23 0%, #0c0f14 65%);
    border: 1px solid #2d1f4e;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.ann-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #7c3aed18 0%, transparent 70%);
    border-radius: 50%;
}
.ann-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 200px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, #ec489918 0%, transparent 70%);
    border-radius: 50%;
}
.ann-header h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #a78bfa;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.ann-header p {
    color: #8892a4;
    font-size: 0.92rem;
    margin: 0;
}
.ann-badge {
    display: inline-block;
    background: #1e1040;
    border: 1px solid #7c3aed;
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 4px;
    margin-right: 8px;
    margin-top: 10px;
}

/* Text area */
.stTextArea textarea {
    background-color: #13161e !important;
    color: #dde1e7 !important;
    border: 1px solid #2d1f4e !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    caret-color: #7c3aed !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px #7c3aed20 !important;
}

/* Buttons */
.stButton > button {
    background: #7c3aed !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #6d28d9 !important;
}

/* Result cards */
.result-card {
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 18px;
    border: 1px solid;
}
.result-ai    { background: #1a0a0a; border-color: #ef4444; }
.result-human { background: #0a1118; border-color: #22d3ee; }
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.label-ai    { color: #ef4444; }
.label-human { color: #22d3ee; }
.result-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #8892a4;
}

/* Probability bar */
.prob-track {
    background: #1a1f2e;
    border-radius: 6px;
    height: 10px;
    margin: 14px 0 4px;
    overflow: hidden;
}
.prob-fill-ai    { height:100%; background:linear-gradient(90deg,#ef4444,#f87171); border-radius:6px; }
.prob-fill-human { height:100%; background:linear-gradient(90deg,#22d3ee,#67e8f9); border-radius:6px; }

/* Pipeline steps */
.pipeline-step {
    background: #13161e;
    border: 1px solid #2d1f4e;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
}
.pipeline-step .step-label {
    color: #8892a4;
    font-size: 0.68rem;
    margin-bottom: 4px;
}
.pipeline-step .step-val {
    color: #a78bfa;
    font-size: 0.88rem;
    font-weight: 500;
}
.pipeline-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4b3a6e;
    font-size: 1.3rem;
    padding: 0 4px;
}

/* Section labels */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: #4b5563;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1a1f2e;
}

/* Metric boxes */
.metric-box {
    background: #13161e;
    border: 1px solid #2d1f4e;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #a78bfa;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #4b5563;
    margin-top: 4px;
}

/* PCA component table */
.pca-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
}
.pca-table th {
    background: #1a1f2e;
    color: #8892a4;
    font-weight: 500;
    padding: 7px 12px;
    text-align: left;
    border-bottom: 1px solid #2d1f4e;
}
.pca-table td {
    padding: 6px 12px;
    border-bottom: 1px solid #1a1f2e;
    color: #b0b8c8;
}
.pca-table tr:hover td { background: #13161e; }

/* Spinner override */
.stSpinner > div { border-top-color: #7c3aed !important; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
PATHS = {
    "pca":           "bert_pca.pkl",
    "scaler":        "mlp_scaler.pkl",
    "model":         "mlp_model.pkl",
    "label_encoder": "mlp_label_encoder.pkl",
}
BERT_MODEL = "bert-base-uncased"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    errors = []
    result = {}
    for key, path in PATHS.items():
        try:
            if key == "pca":
                result[key] = joblib.load(path)
            else:
                with open(path, "rb") as f:
                    result[key] = pickle.load(f)
        except FileNotFoundError:
            errors.append(f"`{path}` not found")
        except Exception as e:
            errors.append(f"`{path}`: {e}")
    return result, errors

@st.cache_resource(show_spinner=False)
def load_bert():
    try:
        import torch
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        bert      = BertModel.from_pretrained(BERT_MODEL)
        bert.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert   = bert.to(device)
        return tokenizer, bert, device, None
    except Exception as e:
        return None, None, None, str(e)

# ── Embedding + prediction ────────────────────────────────────────────────────
def get_bert_embedding(text, tokenizer, bert, device):
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb.astype(np.float32)

def predict_text(text, artifacts, tokenizer, bert, device):
    # Step 1: BERT
    emb = get_bert_embedding(text, tokenizer, bert, device)  # (1,768)
    # Step 2: PCA
    reduced = artifacts["pca"].transform(emb).astype(np.float32)  # (1,80)
    # Step 3: Scaler
    scaled = artifacts["scaler"].transform(reduced)
    # Step 4: MLP
    mlp        = artifacts["model"]
    pred_int   = mlp.predict(scaled)[0]
    pred_proba = mlp.predict_proba(scaled)[0]
    label_map  = {0: "human", 1: "ai"}
    pred_label = label_map[int(pred_int)]
    prob_ai    = float(pred_proba[1])
    prob_human = float(pred_proba[0])
    confidence = float(np.max(pred_proba))
    return {
        "label":      pred_label,
        "prob_ai":    round(prob_ai,    4),
        "prob_human": round(prob_human, 4),
        "confidence": round(confidence, 4),
        "embedding":  emb[0],          # 768-dim for display
        "reduced":    reduced[0],      # 80-dim PCA
    }

def get_confidence_tier(conf):
    if conf >= 0.90: return "Very High ✅"
    if conf >= 0.75: return "High 🟢"
    if conf >= 0.60: return "Medium 🟡"
    return "Low 🔴"

# ── Chart: PCA component magnitudes ──────────────────────────────────────────
def make_pca_chart(reduced_vec, n=30):
    vals = np.abs(reduced_vec[:n])
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0c0f14")
    ax.set_facecolor("#13161e")
    colors = ["#7c3aed" if v > np.percentile(vals, 70)
              else "#4c1d95" if v > np.percentile(vals, 40)
              else "#2d1f4e" for v in vals]
    ax.bar(range(n), vals, color=colors, edgecolor="none", width=0.7)
    ax.set_xlabel("PCA Component Index", color="#4b5563", fontsize=8)
    ax.set_ylabel("|Magnitude|", color="#4b5563", fontsize=8)
    ax.tick_params(colors="#8892a4", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1a1f2e", linewidth=0.6)
    ax.set_axisbelow(True)
    legend_els = [
        mpatches.Patch(color="#7c3aed", label="High activation"),
        mpatches.Patch(color="#4c1d95", label="Medium"),
        mpatches.Patch(color="#2d1f4e", label="Low"),
    ]
    ax.legend(handles=legend_els, loc="upper right", fontsize=7,
              facecolor="#1a1f2e", edgecolor="#2d1f4e", labelcolor="#8892a4")
    plt.tight_layout(pad=1.0)
    return fig

# ── Chart: BERT embedding distribution ───────────────────────────────────────
def make_embedding_hist(embedding):
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor("#0c0f14")
    ax.set_facecolor("#13161e")
    ax.hist(embedding, bins=50, color="#7c3aed", alpha=0.75, edgecolor="none")
    ax.axvline(np.mean(embedding), color="#ec4899", linewidth=1.5,
               linestyle="--", label=f"mean={np.mean(embedding):.3f}")
    ax.set_xlabel("Activation value", color="#4b5563", fontsize=8)
    ax.set_ylabel("Count", color="#4b5563", fontsize=8)
    ax.tick_params(colors="#8892a4", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1a1f2e", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, facecolor="#1a1f2e",
              edgecolor="#2d1f4e", labelcolor="#8892a4")
    plt.tight_layout(pad=1.0)
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ann-header">
  <h1>🧠 BERT-MLP · AI Text Detector</h1>
  <p>Deep semantic classification via BERT embeddings + PCA + MLP</p>
  <span class="ann-badge">MAGE 170k</span>
  <span class="ann-badge">AUC 0.9751</span>
  <span class="ann-badge">Acc 91.78%</span>
  <span class="ann-badge">256→128→64 · ReLU</span>
</div>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
with st.spinner("Loading model artifacts…"):
    artifacts, load_errors = load_artifacts()

if load_errors:
    for err in load_errors:
        st.error(f"Missing file: {err}")
    st.info("Place `bert_pca.pkl`, `mlp_scaler.pkl`, `mlp_model.pkl`, and "
            "`mlp_label_encoder.pkl` in the same directory as this script.")
    st.stop()

# ── Layout ────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="section-label">Input Text</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        label="",
        height=260,
        placeholder="Paste or type your text here…",
        label_visibility="collapsed",
    )

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        analyse = st.button("Analyse ›", use_container_width=True)
    with info_col:
        if input_text:
            wc = len(input_text.split())
            st.markdown(
                f'<p style="color:#a78bfa; font-family:\'DM Mono\',monospace; '
                f'font-size:0.82rem; margin-top:10px;">word count: {wc}</p>',
                unsafe_allow_html=True,
            )

with col_right:
    st.markdown('<div class="section-label">Model Metrics (Test Set)</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-box"><div class="metric-val">0.9178</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-box"><div class="metric-val">0.9178</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-box"><div class="metric-val">0.9751</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Inference Pipeline</div>', unsafe_allow_html=True)

    pc1, pa1, pc2, pa2, pc3, pa3, pc4 = st.columns([3,1,3,1,3,1,3])
    with pc1:
        st.markdown('<div class="pipeline-step"><div class="step-label">INPUT</div><div class="step-val">Raw Text</div></div>', unsafe_allow_html=True)
    with pa1:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
    with pc2:
        st.markdown('<div class="pipeline-step"><div class="step-label">BERT</div><div class="step-val">768-dim</div></div>', unsafe_allow_html=True)
    with pa2:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
    with pc3:
        st.markdown('<div class="pipeline-step"><div class="step-label">PCA</div><div class="step-val">80-dim</div></div>', unsafe_allow_html=True)
    with pa3:
        st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
    with pc4:
        st.markdown('<div class="pipeline-step"><div class="step-label">MLP</div><div class="step-val">Label</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">CV Performance (5-fold)</div>', unsafe_allow_html=True)
    cv_data = {
        "Fold 1": (0.9181, 0.9754),
        "Fold 2": (0.9181, 0.9754),
        "Fold 3": (0.9189, 0.9767),
        "Fold 4": (0.9162, 0.9742),
        "Fold 5": (0.9157, 0.9752),
    }
    fig_cv, ax_cv = plt.subplots(figsize=(5.5, 2.4))
    fig_cv.patch.set_facecolor("#0c0f14")
    ax_cv.set_facecolor("#13161e")
    folds = list(cv_data.keys())
    accs  = [v[0] for v in cv_data.values()]
    aucs  = [v[1] for v in cv_data.values()]
    x = np.arange(len(folds))
    ax_cv.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#7c3aed", alpha=0.85)
    ax_cv.bar(x + 0.2, aucs, 0.35, label="ROC-AUC", color="#ec4899", alpha=0.85)
    ax_cv.set_ylim(0.88, 0.98)
    ax_cv.set_xticks(x); ax_cv.set_xticklabels(folds, fontsize=7.5)
    ax_cv.tick_params(colors="#8892a4", labelsize=7.5)
    ax_cv.spines[:].set_visible(False)
    ax_cv.yaxis.grid(True, color="#1a1f2e", linewidth=0.6)
    ax_cv.set_axisbelow(True)
    ax_cv.legend(fontsize=7.5, facecolor="#1a1f2e",
                 edgecolor="#2d1f4e", labelcolor="#8892a4")
    plt.tight_layout(pad=1.0)
    st.pyplot(fig_cv, use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
if analyse:
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Loading BERT model (first run may take ~60 seconds)…"):
            tokenizer, bert, device, bert_err = load_bert()

        if bert_err:
            st.error(f"BERT loading failed: {bert_err}")
            st.info("Install dependencies: `pip install torch transformers`")
        else:
            with st.spinner("Running BERT + PCA + MLP inference…"):
                try:
                    result = predict_text(input_text, artifacts, tokenizer, bert, device)
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

            st.markdown("---")
            r1, r2 = st.columns([2, 3], gap="large")

            with r1:
                st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
                is_ai    = result["label"] == "ai"
                card_cls = "result-ai" if is_ai else "result-human"
                lbl_cls  = "label-ai" if is_ai else "label-human"
                icon     = "🤖" if is_ai else "🧑"
                lbl_txt  = "AI-Generated" if is_ai else "Human-Written"
                pct      = int(result["prob_ai"] * 100)
                fill_cls = "prob-fill-ai" if is_ai else "prob-fill-human"
                tier     = get_confidence_tier(result["confidence"])

                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <div class="result-label {lbl_cls}">{icon} {lbl_txt}</div>
                  <div class="result-meta">P(AI) = {result['prob_ai']:.4f} &nbsp;|&nbsp; P(Human) = {result['prob_human']:.4f}</div>
                  <div class="prob-track">
                    <div class="{fill_cls}" style="width:{pct}%"></div>
                  </div>
                  <div class="result-meta">Confidence: {int(result['confidence']*100)}% &nbsp;·&nbsp; {tier}</div>
                </div>
                """, unsafe_allow_html=True)

                # Embedding stats
                emb = result["embedding"]
                st.markdown(f"""
                <div style="background:#13161e; border:1px solid #2d1f4e; border-radius:8px;
                     padding:14px 18px; font-family:'DM Mono',monospace; font-size:0.79rem;
                     color:#8892a4; margin-top:4px;">
                  <div style="margin-bottom:4px; color:#a78bfa">BERT [CLS] Embedding Stats</div>
                  <div>dims &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">768</span></div>
                  <div>pca_out: <span style="color:#dde1e7">80</span></div>
                  <div>mean &nbsp;&nbsp;: <span style="color:#dde1e7">{np.mean(emb):.5f}</span></div>
                  <div>std &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">{np.std(emb):.5f}</span></div>
                  <div>max &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">{np.max(emb):.5f}</span></div>
                  <div>min &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">{np.min(emb):.5f}</span></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">BERT Embedding Distribution</div>', unsafe_allow_html=True)
                st.pyplot(make_embedding_hist(emb), use_container_width=True)

            with r2:
                st.markdown('<div class="section-label">PCA Feature Activations (first 30 components)</div>', unsafe_allow_html=True)
                st.pyplot(make_pca_chart(result["reduced"]), use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Top 15 PCA Component Values</div>', unsafe_allow_html=True)

                reduced = result["reduced"]
                top15_idx = np.argsort(np.abs(reduced))[::-1][:15]
                rows_html = ""
                for rank, idx in enumerate(top15_idx, 1):
                    val   = reduced[idx]
                    mag   = abs(val)
                    bar_w = int((mag / (max(abs(reduced)) + 1e-9)) * 120)
                    color = "#7c3aed" if mag > np.percentile(abs(reduced), 80) else "#4c1d95"
                    sign  = "+" if val >= 0 else "−"
                    rows_html += f"""
                    <tr>
                      <td style="color:#8892a4">#{rank}</td>
                      <td>PC-{idx:02d}</td>
                      <td style="color:#a78bfa">{sign}{abs(val):.5f}</td>
                      <td>
                        <div style="background:#1a1f2e;border-radius:3px;height:5px;width:120px;overflow:hidden">
                          <div style="background:{color};height:100%;width:{bar_w}px"></div>
                        </div>
                      </td>
                    </tr>"""

                st.markdown(f"""
                <table class="pca-table">
                  <thead>
                    <tr><th>Rank</th><th>Component</th><th>Value</th><th>Magnitude</th></tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px; padding-top:16px; border-top:1px solid #1a1f2e;
     font-family:'DM Mono',monospace; font-size:0.72rem; color:#2d374a; text-align:center;">
  BERT-MLP Pipeline · IBA Karachi · AI Course Project · Spring 2026
</div>
""", unsafe_allow_html=True)