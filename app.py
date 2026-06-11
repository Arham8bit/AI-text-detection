"""
app.py — AI Text Detection (DT + MLP) — Hugging Face Space
Two tabs: Decision Tree | MLP/ANN
"""

import re
import numpy as np
import gradio as gr
import joblib
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────
CONTRACTIONS = {
    "ain't","aren't","can't","couldn't","didn't","doesn't","don't","hadn't",
    "hasn't","haven't","he'd","he'll","he's","i'd","i'll","i'm","i've",
    "isn't","it'd","it'll","it's","let's","mightn't","mustn't","needn't",
    "shan't","she'd","she'll","she's","shouldn't","that's","there's","they'd",
    "they'll","they're","they've","wasn't","we'd","we'll","we're","we've",
    "weren't","what'll","what're","what's","what've","where's","who'd","who'll",
    "who're","who's","who've","why's","won't","wouldn't","you'd","you'll",
    "you're","you've"
}
TRANSITIONS = {
    "however","therefore","moreover","furthermore","consequently","nevertheless",
    "nonetheless","additionally","subsequently","meanwhile","accordingly",
    "alternatively","conversely","similarly","likewise","thus","hence","besides",
    "otherwise","instead","indeed","specifically","notably","importantly",
    "although","whereas","since","because","unless","until","despite",
    "provided","given","considering"
}
PASSIVE_RE = re.compile(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE)
VOWEL_RE   = re.compile(r"[aeiouy]+", re.IGNORECASE)

def syllable_count(word):
    word = word.rstrip("e")
    return max(1, len(VOWEL_RE.findall(word)))

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (DT)
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "avg_sent_len","std_sent_len","avg_word_len","std_word_len","ttr",
    "comma_count","period_count","exclaim_count","question_count",
    "sent_burstiness","para_len_variance","hapax_ratio",
    "contraction_rate","poly_ratio","passive_ratio","transition_density",
]

def extract_features(text):
    text  = str(text)
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return [0.0] * 16

    sentences    = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    n_sents      = len(sent_lengths)

    avg_sent_len = float(np.mean(sent_lengths)) if n_sents else 0.0
    std_sent_len = float(np.std(sent_lengths))  if n_sents else 0.0

    word_lengths = [len(w) for w in words]
    avg_word_len = float(np.mean(word_lengths))
    std_word_len = float(np.std(word_lengths))

    ttr           = len(set(w.lower() for w in words)) / n_words
    comma_count   = text.count(",")
    period_count  = text.count(".")
    exclaim_count = text.count("!")
    question_count= text.count("?")

    sent_burstiness = (std_sent_len / avg_sent_len) if (n_sents > 1 and avg_sent_len > 0) else 0.0

    paragraphs        = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    para_lengths      = [len(p.split()) for p in paragraphs]
    para_len_variance = float(np.var(para_lengths)) if len(para_lengths) > 1 else 0.0

    freq        = Counter(w.lower() for w in words)
    hapax_ratio = sum(1 for v in freq.values() if v == 1) / n_words

    contraction_rate   = sum(1 for w in words if w.lower().replace("'","'") in CONTRACTIONS) / n_words
    poly_ratio         = sum(1 for w in words if syllable_count(w) >= 3) / n_words
    passive_ratio      = len(PASSIVE_RE.findall(text)) / n_sents if n_sents else 0.0
    transition_density = sum(1 for w in words if w.lower() in TRANSITIONS) / n_words

    return [
        avg_sent_len, std_sent_len, avg_word_len, std_word_len, ttr,
        comma_count, period_count, exclaim_count, question_count,
        sent_burstiness, para_len_variance, hapax_ratio,
        contraction_rate, poly_ratio, passive_ratio, transition_density,
    ]

# ─────────────────────────────────────────────────────────────
# LOAD DT MODEL
# ─────────────────────────────────────────────────────────────
print("Loading DT model …")
dt_bundle  = joblib.load("DT/dt_model.pkl")
dt_model   = dt_bundle["model"]
dt_scaler  = dt_bundle["scaler"]
dt_metrics = dt_bundle.get("test_metrics", {})
print("  DT model loaded.")

# ─────────────────────────────────────────────────────────────
# LOAD MLP ARTIFACTS
# ─────────────────────────────────────────────────────────────
print("Loading MLP artifacts …")
with open("MLP/mlp_model.pkl",         "rb") as f: mlp_model = pickle.load(f)
with open("MLP/mlp_scaler.pkl",        "rb") as f: mlp_scaler = pickle.load(f)
with open("MLP/mlp_label_encoder.pkl", "rb") as f: mlp_le = pickle.load(f)
pca = joblib.load("MLP/bert_pca.pkl")
print("  MLP artifacts loaded.")

# ─────────────────────────────────────────────────────────────
# LOAD BERT (lazy, cached)
# ─────────────────────────────────────────────────────────────
_bert_loaded = False
tokenizer = None
bert       = None
device     = None

def load_bert():
    global _bert_loaded, tokenizer, bert, device
    if _bert_loaded:
        return
    print("Loading BERT … (first call, may take ~60s)")
    from transformers import BertTokenizer, BertModel
    import torch
    tokenizer    = BertTokenizer.from_pretrained("bert-base-uncased")
    bert         = BertModel.from_pretrained("bert-base-uncased")
    bert.eval()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert         = bert.to(device)
    _bert_loaded = True
    print("  BERT ready.")

def get_bert_embedding(text):
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def confidence_tier(c):
    if c >= 0.90: return "Very High"
    if c >= 0.75: return "High"
    if c >= 0.60: return "Medium"
    return "Low"

def make_feature_chart(feature_values, importances, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # Left: Feature importance (model-wide)
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    sorted_idx = np.argsort(importances)
    colors = ["#7c3aed" if importances[i] > np.median(importances) else "#a78bfa"
              for i in sorted_idx]
    ax1.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors, edgecolor="none")
    ax1.set_yticks(range(len(sorted_idx)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx],
                        fontsize=8, color="white")
    ax1.set_xlabel("Importance", color="white", fontsize=9)
    ax1.set_title("Feature Importance (Model-Wide)", color="white", fontsize=10, fontweight="bold")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # Right: This sample's feature values
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    vals   = np.array(feature_values)
    norm   = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    colors2 = ["#f97316" if v > 0.6 else "#38bdf8" for v in norm]
    ax2.barh(feature_names, norm, color=colors2, edgecolor="none")
    ax2.set_xlabel("Normalised Value", color="white", fontsize=9)
    ax2.set_title("This Sample — Feature Values", color="white", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="white", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    plt.tight_layout(pad=2)
    return fig

def make_pca_chart(pca_vector):
    top_n  = 30
    vals   = pca_vector[:top_n]
    mags   = np.abs(vals)
    norm   = mags / (mags.max() + 1e-8)
    colors = ["#7c3aed" if n > 0.6 else "#a78bfa" if n > 0.3 else "#ddd6fe"
              for n in norm]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    # Left: PCA bar chart
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    ax1.bar(range(top_n), mags, color=colors, edgecolor="none")
    ax1.set_xlabel("PCA Component Index", color="white", fontsize=9)
    ax1.set_ylabel("Magnitude", color="white", fontsize=9)
    ax1.set_title("PCA Feature Activations (First 30)", color="white",
                  fontsize=10, fontweight="bold")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # Right: top 15 PCA table
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.axis("off")
    top15_idx  = np.argsort(mags)[::-1][:15]
    table_data = [[f"#{i+1}", f"PC-{idx:02d}", f"{vals[idx]:+.5f}", f"{mags[idx]:.5f}"]
                  for i, idx in enumerate(top15_idx)]
    table = ax2.table(
        cellText=table_data,
        colLabels=["Rank", "Component", "Value", "Magnitude"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#1c2128" if r % 2 == 0 else "#161b22")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#30363d")
    ax2.set_title("Top 15 PCA Component Values", color="white",
                  fontsize=10, fontweight="bold")

    plt.tight_layout(pad=2)
    return fig

# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────
def predict_dt(text):
    if not text or len(text.split()) < 10:
        return "⚠️ Please enter at least 10 words.", None, ""

    feats  = extract_features(text)
    X      = np.array([feats])
    X_s    = dt_scaler.transform(X)

    prob   = dt_model.predict_proba(X_s)[0]
    THRESHOLD = 0.70
    p_ai   = float(prob[1])
    p_hum  = float(prob[0])
    pred   = "AI-Generated" if p_ai >= THRESHOLD else "Human-Written"
    conf   = max(p_ai, p_hum)
    tier   = confidence_tier(conf)

    icon   = "🤖" if pred == "AI-Generated" else "🧑"
    label_color = "#f97316" if pred == "AI-Generated" else "#22c55e"

    result_md = f"""
### {icon} Prediction: **{pred}**

| | |
|---|---|
| P(AI) | `{p_ai:.4f}` |
| P(Human) | `{p_hum:.4f}` |
| Confidence | `{conf:.2%}` — **{tier}** |

---
**Model Metrics (Test Set)**

| Accuracy | F1 Score | ROC AUC |
|---|---|---|
| `{dt_metrics.get('accuracy', 'N/A')}` | `{dt_metrics.get('f1_score', 'N/A')}` | `{dt_metrics.get('roc_auc', 'N/A')}` |
"""

    importances = np.array(dt_model.feature_importances_)
    fig = make_feature_chart(feats, importances, FEATURE_COLS)

    features_md = "**Extracted Features**\n\n| Feature | Value |\n|---|---|\n"
    for name, val in zip(FEATURE_COLS, feats):
        features_md += f"| {name} | `{val:.4f}` |\n"

    return result_md, fig, features_md


def predict_mlp(text):
    if not text or len(text.split()) < 10:
        return "⚠️ Please enter at least 10 words.", None, ""

    load_bert()

    embedding = get_bert_embedding(text)          # (1, 768)
    reduced   = pca.transform(embedding).astype(np.float32)  # (1, 80)
    scaled    = mlp_scaler.transform(reduced)

    pred_int   = mlp_model.predict(scaled)[0]
    pred_proba = mlp_model.predict_proba(scaled)[0]
    label_map  = {0: "Human-Written", 1: "AI-Generated"}
    pred_label = label_map[int(pred_int)]

    p_ai    = float(pred_proba[1])
    p_hum   = float(pred_proba[0])
    conf    = float(np.max(pred_proba))
    tier    = confidence_tier(conf)
    icon    = "🤖" if "AI" in pred_label else "🧑"

    emb_flat = embedding[0]
    result_md = f"""
### {icon} Prediction: **{pred_label}**

| | |
|---|---|
| P(AI) | `{p_ai:.4f}` |
| P(Human) | `{p_hum:.4f}` |
| Confidence | `{conf:.2%}` — **{tier}** |

---
**BERT [CLS] Embedding Stats**

| dims | pca_out | mean | std | max | min |
|---|---|---|---|---|---|
| `768` | `80` | `{emb_flat.mean():.5f}` | `{emb_flat.std():.5f}` | `{emb_flat.max():.5f}` | `{emb_flat.min():.5f}` |
"""

    pca_vec = reduced[0]
    fig     = make_pca_chart(pca_vec)

    dist_md = f"""**BERT Embedding Distribution**

- Mean : `{emb_flat.mean():.5f}`
- Std  : `{emb_flat.std():.5f}`
- Max  : `{emb_flat.max():.5f}`
- Min  : `{emb_flat.min():.5f}`
"""

    return result_md, fig, dist_md


# ─────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────
css = """
body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; }
.tab-nav button { background: #161b22 !important; color: #8b949e !important; border: 1px solid #30363d !important; }
.tab-nav button.selected { background: #1f6feb !important; color: white !important; }
textarea, .gr-textbox { background: #161b22 !important; color: #e6edf3 !important; border: 1px solid #30363d !important; }
.gr-button-primary { background: #1f6feb !important; color: white !important; border: none !important; }
.gr-button-primary:hover { background: #388bfd !important; }
.gr-markdown { color: #e6edf3 !important; }
.gr-panel { background: #161b22 !important; border: 1px solid #30363d !important; }
"""

HEADER = """
# 🤖 AI Text Detection
### Stylometric Decision Tree  ·  BERT + MLP/ANN
*Paste any text and detect whether it was written by a human or generated by AI.*
"""

DT_DESC = """
**Decision Tree** — Uses 16 hand-crafted stylometric features (sentence rhythm, vocabulary richness, punctuation patterns, passive voice, etc.)

- MAGE 170k dataset · AUC 0.8882 · depth-15 · gini · Threshold 0.70
"""

MLP_DESC = """
**MLP / ANN** — Uses BERT [CLS] embeddings (768-dim) → PCA (80-dim) → Multilayer Perceptron

- BERT base-uncased · 80 PCA components · ⚠️ *First run loads BERT (~60s)*
"""

with gr.Blocks(css=css, title="AI Text Detection") as demo:
    gr.Markdown(HEADER)

    with gr.Tabs():

        # ── TAB 1: Decision Tree ──────────────────────────────────────────────
        with gr.Tab("🌳 Decision Tree"):
            gr.Markdown(DT_DESC)
            with gr.Row():
                with gr.Column(scale=1):
                    dt_input  = gr.Textbox(
                        label="Input Text",
                        placeholder="Paste your text here (minimum 10 words)…",
                        lines=10
                    )
                    dt_btn    = gr.Button("Analyse ›", variant="primary")
                    dt_feats  = gr.Markdown(label="Features")
                with gr.Column(scale=1):
                    dt_result = gr.Markdown(label="Prediction")
            dt_plot = gr.Plot(label="Feature Analysis")

            dt_btn.click(
                fn=predict_dt,
                inputs=dt_input,
                outputs=[dt_result, dt_plot, dt_feats]
            )

        # ── TAB 2: MLP / ANN ─────────────────────────────────────────────────
        with gr.Tab("🧠 MLP / ANN (BERT)"):
            gr.Markdown(MLP_DESC)
            with gr.Row():
                with gr.Column(scale=1):
                    mlp_input  = gr.Textbox(
                        label="Input Text",
                        placeholder="Paste your text here (minimum 10 words)…",
                        lines=10
                    )
                    mlp_btn    = gr.Button("Analyse ›", variant="primary")
                    mlp_stats  = gr.Markdown(label="Embedding Stats")
                with gr.Column(scale=1):
                    mlp_result = gr.Markdown(label="Prediction")
            mlp_plot = gr.Plot(label="PCA + Embedding Analysis")

            mlp_btn.click(
                fn=predict_mlp,
                inputs=mlp_input,
                outputs=[mlp_result, mlp_plot, mlp_stats]
            )

    gr.Markdown("""
---
*Built by Arham Awan · Muhammad Ismail · Karan Kumar · Mustafa Khan*
""")

demo.launch()
