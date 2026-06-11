"""
app.py — Combined AI Text Detector (DT + MLP)
Streamlit Community Cloud deployment
Run: streamlit run app.py
"""

import re
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #0c0f14; color: #dde1e7; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    padding: 8px 24px;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

/* ── DT styles ── */
.dt-header {
    background: linear-gradient(135deg, #1a2744 0%, #0f1117 60%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.dt-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #93c5fd;
    margin: 0 0 6px 0;
}
.dt-header p { color: #94a3b8; font-size: 0.92rem; margin: 0; }
.dt-badge {
    display: inline-block;
    background: #1e3a5f;
    border: 1px solid #3b82f6;
    color: #93c5fd;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 4px;
    margin-right: 8px;
    margin-top: 10px;
}

/* ── ANN styles ── */
.ann-header {
    background: linear-gradient(135deg, #130f23 0%, #0c0f14 65%);
    border: 1px solid #2d1f4e;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.ann-header h1 {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 500;
    color: #a78bfa;
    margin: 0 0 6px 0;
}
.ann-header p { color: #8892a4; font-size: 0.92rem; margin: 0; }
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

/* ── Shared UI ── */
.stTextArea textarea {
    background-color: #13161e !important;
    color: #dde1e7 !important;
    border: 1px solid #2a3a5c !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}
.stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
}
.result-card { border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; border: 1px solid; }
.result-ai    { background: #1a0a0a; border-color: #ef4444; }
.result-human-dt { background: #0a1a0f; border-color: #22c55e; }
.result-human-ann { background: #0a1118; border-color: #22d3ee; }
.result-label { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }
.label-ai    { color: #ef4444; }
.label-human-dt  { color: #22c55e; }
.label-human-ann { color: #22d3ee; }
.result-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #94a3b8; }
.prob-track { background: #1e293b; border-radius: 6px; height: 10px; margin: 14px 0 4px; overflow: hidden; }
.prob-fill-ai    { height:100%; background:linear-gradient(90deg,#ef4444,#f87171); border-radius:6px; }
.prob-fill-human-dt  { height:100%; background:linear-gradient(90deg,#22c55e,#4ade80); border-radius:6px; }
.prob-fill-human-ann { height:100%; background:linear-gradient(90deg,#22d3ee,#67e8f9); border-radius:6px; }
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 600; color: #64748b;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid #1e293b;
}
.metric-box { background: #161c2d; border: 1px solid #2a3a5c; border-radius: 8px; padding: 14px 18px; text-align: center; }
.metric-val-dt  { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #93c5fd; }
.metric-val-ann { font-family: 'DM Mono', monospace; font-size: 1.5rem; font-weight: 500; color: #a78bfa; }
.metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
.feat-table { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
.feat-table th { background:#1e293b; color:#94a3b8; font-weight:600; padding:8px 12px; text-align:left; border-bottom:1px solid #2a3a5c; }
.feat-table td { padding:7px 12px; border-bottom:1px solid #1e293b; color:#cbd5e1; }
.feat-rank { display:inline-block; background:#1e3a5f; color:#93c5fd; border-radius:3px; padding:1px 6px; font-size:0.72rem; margin-right:6px; }
.pca-table { width:100%; border-collapse:collapse; font-family:'DM Mono',monospace; font-size:0.78rem; }
.pca-table th { background:#1a1f2e; color:#8892a4; font-weight:500; padding:7px 12px; text-align:left; border-bottom:1px solid #2d1f4e; }
.pca-table td { padding:6px 12px; border-bottom:1px solid #1a1f2e; color:#b0b8c8; }
.pipeline-step { background:#13161e; border:1px solid #2d1f4e; border-radius:8px; padding:12px 16px; text-align:center; font-family:'DM Mono',monospace; font-size:0.78rem; }
.pipeline-step .step-label { color:#8892a4; font-size:0.68rem; margin-bottom:4px; }
.pipeline-step .step-val   { color:#a78bfa; font-size:0.88rem; font-weight:500; }
.pipeline-arrow { display:flex; align-items:center; justify-content:center; color:#4b3a6e; font-size:1.3rem; padding:0 4px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ── DT IMPORTS & HELPERS ──────────────────────────────────────────────────────
# =============================================================================
import joblib
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import nltk
from collections import Counter

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

TREE_UNDEFINED = -2

FEATURE_COLS = [
    "avg_sent_len","std_sent_len","avg_word_len","std_word_len","ttr",
    "comma_count","period_count","exclaim_count","question_count",
    "sent_burstiness","para_len_variance","hapax_ratio",
    "contraction_rate","poly_ratio","passive_ratio","transition_density",
]
FEATURE_DESC = {
    "hapax_ratio":        "Fraction of words that appear exactly once",
    "period_count":       "Count of sentence-ending periods",
    "sent_burstiness":    "Irregularity in sentence length",
    "ttr":                "Type-Token Ratio — vocabulary breadth",
    "avg_sent_len":       "Mean token count per sentence",
    "std_sent_len":       "Std deviation of sentence lengths",
    "transition_density": "Rate of discourse connectors",
    "avg_word_len":       "Mean character count per word",
    "passive_ratio":      "Fraction of sentences using passive voice",
    "poly_ratio":         "Fraction of words with 3+ syllables",
    "std_word_len":       "Std dev of word character lengths",
    "comma_count":        "Count of commas",
    "contraction_rate":   "Rate of contractions (don't, it's…)",
    "para_len_variance":  "Variance in paragraph word counts",
    "exclaim_count":      "Count of exclamation marks",
    "question_count":     "Count of question marks",
}
CONTRACTIONS = {
    "ain't","aren't","can't","couldn't","didn't","doesn't","don't","hadn't",
    "hasn't","haven't","he'd","he'll","he's","i'd","i'll","i'm","i've",
    "isn't","it's","let's","mightn't","mustn't","shan't","she'd","she'll",
    "she's","shouldn't","that's","there's","they'd","they'll","they're",
    "they've","wasn't","we'd","we'll","we're","we've","weren't","what'll",
    "what're","what's","what've","where's","who'd","who'll","who're","who's",
    "who've","won't","wouldn't","you'd","you'll","you're","you've",
}
TRANSITION_WORDS = {
    "however","therefore","furthermore","moreover","nevertheless","consequently",
    "additionally","alternatively","meanwhile","subsequently","accordingly",
    "nonetheless","likewise","conversely","thus","hence","besides","instead",
    "otherwise","similarly","finally","firstly","secondly","thirdly",
    "in addition","in contrast","in conclusion","for example","for instance",
    "as a result","on the other hand","in other words","that is","in summary",
    "to summarize","in fact","indeed","certainly","notably","importantly",
}
BE_FORMS = {"is","are","was","were","be","been","being","am"}
DT_THRESHOLD = 0.70
DT_MIN_WORDS = 50

def _count_syllables(word):
    word   = word.lower().strip(".,!?;:\"'")
    vowels = re.findall(r"[aeiouy]+", word)
    count  = len(vowels)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def _split_sentences(text):
    try:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def _split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def extract_dt_features(text):
    words = text.split()
    if len(words) < DT_MIN_WORDS:
        return None
    sentences  = _split_sentences(text)
    paragraphs = _split_paragraphs(text)
    sent_lens  = [len(s.split()) for s in sentences] if sentences else [len(words)]
    word_lens  = [len(w.strip(".,!?;:\"'()")) for w in words if w.strip(".,!?;:\"'()")]
    avg_sent_len = float(np.mean(sent_lens))
    std_sent_len = float(np.std(sent_lens))
    avg_word_len = float(np.mean(word_lens)) if word_lens else 0.0
    std_word_len = float(np.std(word_lens))  if word_lens else 0.0
    words_lower  = [w.lower().strip(".,!?;:\"'()") for w in words]
    ttr          = len(set(words_lower)) / len(words_lower) if words_lower else 0.0
    comma_count    = float(text.count(","))
    period_count   = float(text.count("."))
    exclaim_count  = float(text.count("!"))
    question_count = float(text.count("?"))
    denom          = avg_sent_len + std_sent_len
    sent_burstiness = (std_sent_len - avg_sent_len) / denom if denom != 0 else 0.0
    para_lens       = [len(p.split()) for p in paragraphs] if paragraphs else [len(words)]
    para_len_variance = float(np.var(para_lens))
    freq        = Counter(words_lower)
    hapax_ratio = sum(1 for v in freq.values() if v == 1) / len(words_lower)
    contraction_rate = sum(1 for w in words_lower if w in CONTRACTIONS) / len(words_lower)
    poly_ratio   = sum(1 for w in words_lower if _count_syllables(w) >= 3) / len(words_lower)
    passive_count = 0
    for sent in sentences:
        sent_words = sent.lower().split()
        for i, w in enumerate(sent_words[:-1]):
            if w in BE_FORMS:
                nxt = sent_words[i+1].strip(".,!?;:'\"")
                if re.search(r"(ed|en|t)$", nxt):
                    passive_count += 1
                    break
    passive_ratio = passive_count / len(sentences) if sentences else 0.0
    text_lower     = text.lower()
    transition_hits = sum(1 for t in TRANSITION_WORDS if re.search(r'\b' + re.escape(t) + r'\b', text_lower))
    transition_density = transition_hits / len(sentences) if sentences else 0.0
    return {
        "avg_sent_len": round(avg_sent_len, 6), "std_sent_len": round(std_sent_len, 6),
        "avg_word_len": round(avg_word_len, 6), "std_word_len": round(std_word_len, 6),
        "ttr": round(ttr, 6), "comma_count": round(comma_count, 6),
        "period_count": round(period_count, 6), "exclaim_count": round(exclaim_count, 6),
        "question_count": round(question_count, 6), "sent_burstiness": round(sent_burstiness, 6),
        "para_len_variance": round(para_len_variance, 6), "hapax_ratio": round(hapax_ratio, 6),
        "contraction_rate": round(contraction_rate, 6), "poly_ratio": round(poly_ratio, 6),
        "passive_ratio": round(passive_ratio, 6), "transition_density": round(transition_density, 6),
    }

@st.cache_resource
def load_dt_model():
    try:
        return joblib.load("DT/dt_model.pkl"), None
    except Exception as e:
        return None, str(e)

def make_dt_importance_chart(bundle, highlight_features=None):
    importances = bundle["model"].feature_importances_
    imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161c2d")
    colors = []
    for feat in imp.index:
        if highlight_features and feat in highlight_features[:3]:   colors.append("#3b82f6")
        elif highlight_features and feat in highlight_features[3:6]: colors.append("#1d4ed8")
        else:                                                         colors.append("#1e3a5f")
    bars = ax.barh(imp.index, imp.values, color=colors, height=0.65, edgecolor="none")
    for bar, val in zip(bars, imp.values):
        if val > 0.01:
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", ha="left", color="#94a3b8", fontsize=7.5, fontfamily="monospace")
    ax.set_xlabel("Gini Importance", color="#64748b", fontsize=9)
    ax.tick_params(axis="both", colors="#94a3b8", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="#1e293b", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)
    legend_els = [
        mpatches.Patch(color="#3b82f6", label="Top 3 features"),
        mpatches.Patch(color="#1d4ed8", label="Ranks 4–6"),
        mpatches.Patch(color="#1e3a5f", label="Other features"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=7.5,
              facecolor="#1e293b", edgecolor="#2a3a5c", labelcolor="#94a3b8")
    plt.tight_layout(pad=1.2)
    return fig

def make_sample_radar(features):
    keys   = ["hapax_ratio","ttr","sent_burstiness","passive_ratio","poly_ratio","contraction_rate","transition_density"]
    labels = ["Hapax","TTR","Burstiness","Passive","Polysyll","Contraction","Transition"]
    vals   = [abs(features.get(k, 0)) for k in keys]
    max_v  = max(vals) if max(vals) > 0 else 1
    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161c2d")
    bar_colors = ["#3b82f6" if v/max_v > 0.6 else "#1d4ed8" if v/max_v > 0.3 else "#1e3a5f" for v in vals]
    ax.bar(labels, [v/max_v for v in vals], color=bar_colors, edgecolor="none", width=0.55)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Relative value", color="#64748b", fontsize=8)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=7.5, rotation=15)
    ax.tick_params(axis="y", colors="#64748b", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1e293b", linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.0)
    return fig

def make_tree_path_viz(model, feat_scaled, feature_cols):
    TREE_LEAF = -1
    tree        = model.tree_
    path_sparse = model.decision_path(feat_scaled)
    path_nodes  = path_sparse.indices.tolist()
    n_steps     = len(path_nodes)
    PATH_X=6.0; LEVEL_H=2.6; STUB_OFF=4.5
    NW,NH=4.4,1.05; SW,SH=2.6,0.70; LW,LH=4.4,1.05
    fig_w=13.0; fig_h=(n_steps-1)*LEVEL_H+2.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0c111b"); ax.set_facecolor("#0c111b")
    ax.set_xlim(0,12); ax.set_ylim(-0.4, fig_h-0.2)
    ax.invert_yaxis(); ax.axis("off")
    def node_y(s): return s * LEVEL_H
    def draw_rect(cx,cy,w,h,fc,ec,lw=1.5,zorder=3):
        ax.add_patch(mpatches.FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.07",
            facecolor=fc,edgecolor=ec,linewidth=lw,zorder=zorder))
    def draw_arc(x1,y1c,h1,x2,y2c,h2,color,rad=0.0,lw=1.6):
        ax.annotate("",xy=(x2,y2c-h2/2-0.05),xytext=(x1,y1c+h1/2+0.05),
            arrowprops=dict(arrowstyle="-|>",color=color,lw=lw,mutation_scale=10,
            connectionstyle=f"arc3,rad={rad}"),zorder=2)
    def arc_label(x,y,text,color):
        ax.text(x,y,text,ha="center",va="center",fontsize=7.5,color=color,
            fontfamily="monospace",fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18",fc="#0c111b",ec="none"),zorder=5)
    for step, node_id in enumerate(path_nodes):
        cy      = node_y(step)
        is_leaf = (tree.feature[node_id] == TREE_UNDEFINED)
        if is_leaf:
            counts=tree.value[node_id][0]; pred_class=int(np.argmax(counts))
            human_n=int(counts[0]); ai_n=int(counts[1]); total_n=human_n+ai_n
            purity=round(max(counts)/total_n*100,1) if total_n else 0
            is_ai=pred_class==1
            ec="#ef4444" if is_ai else "#22c55e"; fc="#2d0a0a" if is_ai else "#0a2318"
            txt_col="#ef4444" if is_ai else "#22c55e"
            icon="🤖" if is_ai else "🧑"; lbl="AI-Generated" if is_ai else "Human-Written"
            draw_rect(PATH_X,cy,LW,LH,fc,ec,lw=2.2)
            ax.text(PATH_X,cy-0.30,"LEAF · FINAL PREDICTION",ha="center",va="center",fontsize=6,color="#475569",fontfamily="monospace")
            ax.text(PATH_X,cy+0.02,f"{icon}  {lbl}",ha="center",va="center",fontsize=10.5,color=txt_col,fontweight="bold",fontfamily="monospace")
            ax.text(PATH_X,cy+0.33,f"human={human_n}   ai={ai_n}   total={total_n}   ({purity}% pure)",ha="center",va="center",fontsize=6.5,color="#475569",fontfamily="monospace")
        else:
            feat_idx=tree.feature[node_id]; feat_name=feature_cols[feat_idx]
            threshold=tree.threshold[node_id]; actual_val=float(feat_scaled[0,feat_idx])
            next_id=path_nodes[step+1] if step+1 < n_steps else None
            went_left=None
            if next_id is not None: went_left=(next_id==tree.children_left[node_id])
            draw_rect(PATH_X,cy,NW,NH,"#111827","#3b82f6")
            ax.text(PATH_X,cy-0.30,feat_name,ha="center",va="center",fontsize=9.5,color="#93c5fd",fontweight="bold",fontfamily="monospace")
            ax.text(PATH_X,cy+0.04,f"≤  {threshold:.5f}",ha="center",va="center",fontsize=8,color="#94a3b8",fontfamily="monospace")
            ax.text(PATH_X,cy+0.33,f"your value: {actual_val:.5f}",ha="center",va="center",fontsize=7.5,color="#64748b",fontfamily="monospace")
            if next_id is not None:
                next_cy=node_y(step+1)
                stub_id=tree.children_right[node_id] if went_left else tree.children_left[node_id]
                stub_samples=int(tree.n_node_samples[stub_id])
                stub_is_leaf=(tree.feature[stub_id]==TREE_UNDEFINED)
                stub_kind="leaf" if stub_is_leaf else "subtree"
                if went_left:
                    stub_x=PATH_X+STUB_OFF; path_rad=0.12; stub_rad=-0.20
                    true_lbl_x=PATH_X-1.2; false_lbl_x=PATH_X+STUB_OFF/2+0.5
                    true_lbl_y=(cy+next_cy)/2; false_lbl_y=(cy+next_cy)/2-0.2
                else:
                    stub_x=PATH_X-STUB_OFF; path_rad=-0.12; stub_rad=0.20
                    true_lbl_x=PATH_X-STUB_OFF/2-0.5; false_lbl_x=PATH_X+1.2
                    true_lbl_y=(cy+next_cy)/2-0.2; false_lbl_y=(cy+next_cy)/2
                stub_cy=next_cy
                draw_rect(stub_x,stub_cy,SW,SH,fc="#0f172a",ec="#2a3a5c",lw=1.1)
                ax.text(stub_x,stub_cy-0.13,stub_kind,ha="center",va="center",fontsize=7.5,color="#3b4f6b",fontfamily="monospace")
                ax.text(stub_x,stub_cy+0.15,f"{stub_samples} samples",ha="center",va="center",fontsize=7,color="#2a3a5c",fontfamily="monospace")
                draw_arc(PATH_X,cy,NH,PATH_X,next_cy,NH if not(step+1==n_steps-1) else LH,color="#22c55e",rad=path_rad,lw=2.0)
                draw_arc(PATH_X,cy,NH,stub_x,stub_cy,SH,color="#2a3a5c",rad=stub_rad,lw=1.3)
                arc_label(true_lbl_x,true_lbl_y,"True (≤)","#22c55e")
                arc_label(false_lbl_x,false_lbl_y,"False (>)","#475569")
    legend_items=[
        mpatches.Patch(facecolor="#22c55e",edgecolor="none",label="Taken branch  (True ≤ threshold)"),
        mpatches.Patch(facecolor="#2a3a5c",edgecolor="none",label="Skipped branch  (collapsed stub)"),
        mpatches.Patch(facecolor="#3b82f6",edgecolor="none",label="Decision node"),
    ]
    ax.legend(handles=legend_items,loc="lower center",bbox_to_anchor=(0.5,0.0),
              fontsize=8,ncol=3,facecolor="#111827",edgecolor="#2a3a5c",labelcolor="#94a3b8")
    plt.tight_layout(pad=0.6)
    return fig

# =============================================================================
# ── ANN HELPERS ───────────────────────────────────────────────────────────────
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_mlp_artifacts():
    errors = []
    result = {}
    paths  = {"pca":"MLP/bert_pca.pkl","scaler":"MLP/mlp_scaler.pkl",
              "model":"MLP/mlp_model.pkl","label_encoder":"MLP/mlp_label_encoder.pkl"}
    for key, path in paths.items():
        try:
            result[key] = joblib.load(path) if key == "pca" else pickle.load(open(path,"rb"))
        except Exception as e:
            errors.append(f"{path}: {e}")
    return result, errors

@st.cache_resource(show_spinner=False)
def load_bert():
    try:
        import torch
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert      = BertModel.from_pretrained("bert-base-uncased")
        bert.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert   = bert.to(device)
        return tokenizer, bert, device, None
    except Exception as e:
        return None, None, None, str(e)

def get_bert_embedding(text, tokenizer, bert, device):
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

def predict_mlp(text, artifacts, tokenizer, bert, device):
    emb     = get_bert_embedding(text, tokenizer, bert, device)
    reduced = artifacts["pca"].transform(emb).astype(np.float32)
    scaled  = artifacts["scaler"].transform(reduced)
    mlp     = artifacts["model"]
    pred_int   = mlp.predict(scaled)[0]
    pred_proba = mlp.predict_proba(scaled)[0]
    label_map  = {0:"human", 1:"ai"}
    return {
        "label":      label_map[int(pred_int)],
        "prob_ai":    round(float(pred_proba[1]), 4),
        "prob_human": round(float(pred_proba[0]), 4),
        "confidence": round(float(np.max(pred_proba)), 4),
        "embedding":  emb[0],
        "reduced":    reduced[0],
    }

def get_confidence_tier(conf):
    if conf >= 0.90: return "Very High ✅"
    if conf >= 0.75: return "High 🟢"
    if conf >= 0.60: return "Medium 🟡"
    return "Low 🔴"

def make_pca_chart(reduced_vec, n=30):
    vals = np.abs(reduced_vec[:n])
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0c0f14"); ax.set_facecolor("#13161e")
    colors = ["#7c3aed" if v > np.percentile(vals,70) else "#4c1d95" if v > np.percentile(vals,40) else "#2d1f4e" for v in vals]
    ax.bar(range(n), vals, color=colors, edgecolor="none", width=0.7)
    ax.set_xlabel("PCA Component Index", color="#4b5563", fontsize=8)
    ax.set_ylabel("|Magnitude|", color="#4b5563", fontsize=8)
    ax.tick_params(colors="#8892a4", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1a1f2e", linewidth=0.6)
    ax.set_axisbelow(True)
    legend_els=[mpatches.Patch(color="#7c3aed",label="High activation"),
                mpatches.Patch(color="#4c1d95",label="Medium"),
                mpatches.Patch(color="#2d1f4e",label="Low")]
    ax.legend(handles=legend_els, loc="upper right", fontsize=7,
              facecolor="#1a1f2e", edgecolor="#2d1f4e", labelcolor="#8892a4")
    plt.tight_layout(pad=1.0)
    return fig

def make_embedding_hist(embedding):
    fig, ax = plt.subplots(figsize=(6, 2.4))
    fig.patch.set_facecolor("#0c0f14"); ax.set_facecolor("#13161e")
    ax.hist(embedding, bins=50, color="#7c3aed", alpha=0.75, edgecolor="none")
    ax.axvline(np.mean(embedding), color="#ec4899", linewidth=1.5, linestyle="--",
               label=f"mean={np.mean(embedding):.3f}")
    ax.set_xlabel("Activation value", color="#4b5563", fontsize=8)
    ax.set_ylabel("Count", color="#4b5563", fontsize=8)
    ax.tick_params(colors="#8892a4", labelsize=7)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1a1f2e", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, facecolor="#1a1f2e", edgecolor="#2d1f4e", labelcolor="#8892a4")
    plt.tight_layout(pad=1.0)
    return fig

# =============================================================================
# ── TABS ──────────────────────────────────────────────────────────────────────
# =============================================================================
tab1, tab2 = st.tabs(["🌳  Decision Tree", "🧠  MLP / ANN (BERT)"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DECISION TREE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="dt-header">
      <h1>🌳 Decision Tree · AI Text Detector</h1>
      <p>Stylometric analysis using 16 hand-crafted linguistic features</p>
      <span class="dt-badge">MAGE 170k</span>
      <span class="dt-badge">AUC 0.8882</span>
      <span class="dt-badge">Threshold 0.70</span>
      <span class="dt-badge">depth-15 · gini</span>
    </div>
    """, unsafe_allow_html=True)

    dt_bundle, dt_err = load_dt_model()
    if dt_err:
        st.error(f"Could not load DT model: {dt_err}")
        st.stop()

    model_metrics = dt_bundle.get("test_metrics", {})
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Input Text</div>', unsafe_allow_html=True)
        dt_input = st.text_area("", height=260,
            placeholder="Paste or type your text here (minimum 50 words)…",
            label_visibility="collapsed", key="dt_input")
        btn_col, info_col = st.columns([1, 3])
        with btn_col:
            dt_analyse = st.button("Analyse ›", key="dt_btn")
        with info_col:
            if dt_input:
                wc = len(dt_input.split())
                color = "#22c55e" if wc >= DT_MIN_WORDS else "#f59e0b"
                st.markdown(f'<p style="color:{color};font-family:IBM Plex Mono,monospace;font-size:0.82rem;margin-top:10px;">word count: {wc}{"  ✓" if wc >= DT_MIN_WORDS else f"  (need {DT_MIN_WORDS-wc} more)"}</p>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">Model Metrics (Test Set)</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f'<div class="metric-box"><div class="metric-val-dt">{model_metrics.get("accuracy","—")}</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-box"><div class="metric-val-dt">{model_metrics.get("f1_score","—")}</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-box"><div class="metric-val-dt">{model_metrics.get("roc_auc","—")}</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Feature Importance (Model-wide)</div>', unsafe_allow_html=True)
        st.pyplot(make_dt_importance_chart(dt_bundle), use_container_width=True)

    if dt_analyse:
        if not dt_input.strip():
            st.warning("Please enter some text first.")
        else:
            features = extract_dt_features(dt_input)
            if features is None:
                st.warning(f"Text too short — minimum {DT_MIN_WORDS} words required.")
            else:
                model      = dt_bundle["model"]
                scaler     = dt_bundle["scaler"]
                lmap       = dt_bundle["label_map"]
                feat_vec   = np.array([[features[c] for c in FEATURE_COLS]])
                feat_scaled= scaler.transform(feat_vec)
                prob_ai    = float(model.predict_proba(feat_scaled)[0][1])
                prediction = int(prob_ai >= DT_THRESHOLD)
                label      = lmap[prediction]
                confidence = prob_ai if prediction == 1 else 1 - prob_ai

                st.markdown("---")
                r1, r2 = st.columns([2, 3], gap="large")
                with r1:
                    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
                    is_ai    = label == "ai"
                    card_cls = "result-ai" if is_ai else "result-human-dt"
                    lbl_cls  = "label-ai" if is_ai else "label-human-dt"
                    icon     = "🤖" if is_ai else "🧑"
                    lbl_txt  = "AI-Generated" if is_ai else "Human-Written"
                    pct      = int(prob_ai * 100)
                    fill_cls = "prob-fill-ai" if is_ai else "prob-fill-human-dt"
                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                      <div class="result-label {lbl_cls}">{icon} {lbl_txt}</div>
                      <div class="result-meta">P(AI) = {prob_ai:.4f} &nbsp;|&nbsp; threshold = {DT_THRESHOLD}</div>
                      <div class="prob-track"><div class="{fill_cls}" style="width:{pct}%"></div></div>
                      <div class="result-meta">Confidence: {int(confidence*100)}%</div>
                    </div>""", unsafe_allow_html=True)
                    wc    = len(dt_input.split())
                    sents = len(re.findall(r'[.!?]+', dt_input)) or 1
                    st.markdown(f"""
                    <div style="background:#161c2d;border:1px solid #2a3a5c;border-radius:8px;padding:14px 18px;font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#94a3b8;margin-top:4px;">
                      <div>words &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#e2e8f0">{wc}</span></div>
                      <div>sentences : <span style="color:#e2e8f0">{sents}</span></div>
                      <div>hapax_ratio: <span style="color:#93c5fd">{features['hapax_ratio']:.4f}</span></div>
                      <div>ttr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#93c5fd">{features['ttr']:.4f}</span></div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Sample Feature Profile</div>', unsafe_allow_html=True)
                    st.pyplot(make_sample_radar(features), use_container_width=True)

                with r2:
                    st.markdown('<div class="section-label">All 16 Feature Values</div>', unsafe_allow_html=True)
                    importances  = dt_bundle["model"].feature_importances_
                    imp_map      = dict(zip(FEATURE_COLS, importances))
                    sorted_feats = sorted(FEATURE_COLS, key=lambda f: imp_map[f], reverse=True)
                    rows_html = ""
                    for rank, feat in enumerate(sorted_feats, 1):
                        val   = features[feat]
                        imp   = imp_map[feat]
                        desc  = FEATURE_DESC.get(feat, "")
                        bar_w = int(imp * 600)
                        rows_html += f'<tr><td><span class="feat-rank">#{rank}</span>{feat}</td><td style="color:#93c5fd">{val:.5f}</td><td><div style="background:#1e293b;border-radius:3px;height:5px;width:120px;overflow:hidden"><div style="background:#2563eb;height:100%;width:{bar_w}px"></div></div></td><td style="color:#64748b;font-size:0.72rem">{desc[:55]}{"…" if len(desc)>55 else ""}</td></tr>'
                    st.markdown(f'<table class="feat-table"><thead><tr><th>Feature</th><th>Value</th><th>Importance</th><th>Description</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Decision Tree Path</div>', unsafe_allow_html=True)
                path_sparse = model.decision_path(feat_scaled)
                n_visited   = len(path_sparse.indices)
                path_color  = "#ef4444" if is_ai else "#22c55e"
                st.markdown(f'<p style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#64748b;margin-bottom:10px;">The tree visited <span style="color:{path_color};font-weight:600;">{n_visited} nodes</span> ({n_visited-1} decisions + 1 leaf).</p>', unsafe_allow_html=True)
                with st.expander("🌳 Show full decision path", expanded=True):
                    st.pyplot(make_tree_path_viz(model, feat_scaled, FEATURE_COLS), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MLP / ANN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
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

    with st.spinner("Loading MLP artifacts…"):
        mlp_artifacts, mlp_errors = load_mlp_artifacts()

    if mlp_errors:
        for err in mlp_errors: st.error(f"Missing: {err}")
        st.stop()

    col_left2, col_right2 = st.columns([3, 2], gap="large")

    with col_left2:
        st.markdown('<div class="section-label">Input Text</div>', unsafe_allow_html=True)
        ann_input = st.text_area("", height=260,
            placeholder="Paste or type your text here…",
            label_visibility="collapsed", key="ann_input")
        btn_col2, info_col2 = st.columns([1, 3])
        with btn_col2:
            ann_analyse = st.button("Analyse ›", key="ann_btn")
        with info_col2:
            if ann_input:
                wc = len(ann_input.split())
                st.markdown(f'<p style="color:#a78bfa;font-family:DM Mono,monospace;font-size:0.82rem;margin-top:10px;">word count: {wc}</p>', unsafe_allow_html=True)

    with col_right2:
        st.markdown('<div class="section-label">Model Metrics (Test Set)</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown('<div class="metric-box"><div class="metric-val-ann">0.9178</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
        with m2: st.markdown('<div class="metric-box"><div class="metric-val-ann">0.9178</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
        with m3: st.markdown('<div class="metric-box"><div class="metric-val-ann">0.9751</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Inference Pipeline</div>', unsafe_allow_html=True)
        pc1,pa1,pc2,pa2,pc3,pa3,pc4 = st.columns([3,1,3,1,3,1,3])
        with pc1: st.markdown('<div class="pipeline-step"><div class="step-label">INPUT</div><div class="step-val">Raw Text</div></div>', unsafe_allow_html=True)
        with pa1: st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with pc2: st.markdown('<div class="pipeline-step"><div class="step-label">BERT</div><div class="step-val">768-dim</div></div>', unsafe_allow_html=True)
        with pa2: st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with pc3: st.markdown('<div class="pipeline-step"><div class="step-label">PCA</div><div class="step-val">80-dim</div></div>', unsafe_allow_html=True)
        with pa3: st.markdown('<div class="pipeline-arrow">→</div>', unsafe_allow_html=True)
        with pc4: st.markdown('<div class="pipeline-step"><div class="step-label">MLP</div><div class="step-val">Label</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">CV Performance (5-fold)</div>', unsafe_allow_html=True)
        cv_data = {"Fold 1":(0.9181,0.9754),"Fold 2":(0.9181,0.9754),"Fold 3":(0.9189,0.9767),"Fold 4":(0.9162,0.9742),"Fold 5":(0.9157,0.9752)}
        fig_cv, ax_cv = plt.subplots(figsize=(5.5, 2.4))
        fig_cv.patch.set_facecolor("#0c0f14"); ax_cv.set_facecolor("#13161e")
        folds=list(cv_data.keys()); accs=[v[0] for v in cv_data.values()]; aucs=[v[1] for v in cv_data.values()]
        x=np.arange(len(folds))
        ax_cv.bar(x-0.2,accs,0.35,label="Accuracy",color="#7c3aed",alpha=0.85)
        ax_cv.bar(x+0.2,aucs,0.35,label="ROC-AUC",color="#ec4899",alpha=0.85)
        ax_cv.set_ylim(0.88,0.98); ax_cv.set_xticks(x); ax_cv.set_xticklabels(folds,fontsize=7.5)
        ax_cv.tick_params(colors="#8892a4",labelsize=7.5); ax_cv.spines[:].set_visible(False)
        ax_cv.yaxis.grid(True,color="#1a1f2e",linewidth=0.6); ax_cv.set_axisbelow(True)
        ax_cv.legend(fontsize=7.5,facecolor="#1a1f2e",edgecolor="#2d1f4e",labelcolor="#8892a4")
        plt.tight_layout(pad=1.0)
        st.pyplot(fig_cv, use_container_width=True)

    if ann_analyse:
        if not ann_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Loading BERT model (first run ~60 seconds)…"):
                tokenizer, bert_model, device, bert_err = load_bert()
            if bert_err:
                st.error(f"BERT loading failed: {bert_err}")
            else:
                with st.spinner("Running BERT + PCA + MLP inference…"):
                    try:
                        result = predict_mlp(ann_input, mlp_artifacts, tokenizer, bert_model, device)
                    except Exception as e:
                        st.error(f"Inference error: {e}"); st.stop()

                st.markdown("---")
                r1, r2 = st.columns([2, 3], gap="large")
                with r1:
                    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
                    is_ai    = result["label"] == "ai"
                    card_cls = "result-ai" if is_ai else "result-human-ann"
                    lbl_cls  = "label-ai" if is_ai else "label-human-ann"
                    icon     = "🤖" if is_ai else "🧑"
                    lbl_txt  = "AI-Generated" if is_ai else "Human-Written"
                    pct      = int(result["prob_ai"] * 100)
                    fill_cls = "prob-fill-ai" if is_ai else "prob-fill-human-ann"
                    tier     = get_confidence_tier(result["confidence"])
                    st.markdown(f"""
                    <div class="result-card {card_cls}">
                      <div class="result-label {lbl_cls}">{icon} {lbl_txt}</div>
                      <div class="result-meta">P(AI) = {result['prob_ai']:.4f} &nbsp;|&nbsp; P(Human) = {result['prob_human']:.4f}</div>
                      <div class="prob-track"><div class="{fill_cls}" style="width:{pct}%"></div></div>
                      <div class="result-meta">Confidence: {int(result['confidence']*100)}% &nbsp;·&nbsp; {tier}</div>
                    </div>""", unsafe_allow_html=True)
                    emb = result["embedding"]
                    st.markdown(f"""
                    <div style="background:#13161e;border:1px solid #2d1f4e;border-radius:8px;padding:14px 18px;font-family:'DM Mono',monospace;font-size:0.79rem;color:#8892a4;margin-top:4px;">
                      <div style="margin-bottom:4px;color:#a78bfa">BERT [CLS] Embedding Stats</div>
                      <div>dims &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">768</span></div>
                      <div>pca_out: <span style="color:#dde1e7">80</span></div>
                      <div>mean &nbsp;&nbsp;: <span style="color:#dde1e7">{np.mean(emb):.5f}</span></div>
                      <div>std &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">{np.std(emb):.5f}</span></div>
                      <div>max &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">{np.max(emb):.5f}</span></div>
                      <div>min &nbsp;&nbsp;&nbsp;: <span style="color:#dde1e7">{np.min(emb):.5f}</span></div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">BERT Embedding Distribution</div>', unsafe_allow_html=True)
                    st.pyplot(make_embedding_hist(emb), use_container_width=True)

                with r2:
                    st.markdown('<div class="section-label">PCA Feature Activations (first 30 components)</div>', unsafe_allow_html=True)
                    st.pyplot(make_pca_chart(result["reduced"]), use_container_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Top 15 PCA Component Values</div>', unsafe_allow_html=True)
                    reduced   = result["reduced"]
                    top15_idx = np.argsort(np.abs(reduced))[::-1][:15]
                    rows_html = ""
                    for rank, idx in enumerate(top15_idx, 1):
                        val   = reduced[idx]; mag = abs(val)
                        bar_w = int((mag/(max(abs(reduced))+1e-9))*120)
                        color = "#7c3aed" if mag > np.percentile(abs(reduced),80) else "#4c1d95"
                        sign  = "+" if val >= 0 else "−"
                        rows_html += f'<tr><td style="color:#8892a4">#{rank}</td><td>PC-{idx:02d}</td><td style="color:#a78bfa">{sign}{abs(val):.5f}</td><td><div style="background:#1a1f2e;border-radius:3px;height:5px;width:120px;overflow:hidden"><div style="background:{color};height:100%;width:{bar_w}px"></div></div></td></tr>'
                    st.markdown(f'<table class="pca-table"><thead><tr><th>Rank</th><th>Component</th><th>Value</th><th>Magnitude</th></tr></thead><tbody>{rows_html}</tbody></table>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px;padding-top:16px;border-top:1px solid #1e293b;
font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#334155;text-align:center;">
  AI Text Detection · IBA Karachi · AI Course Project · Spring 2026 ·
  Arham Awan · Muhammad Ismail · Karan Kumar · Mustafa Khan
</div>
""", unsafe_allow_html=True)
