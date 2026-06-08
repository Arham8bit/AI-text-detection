import numpy as np
import json
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS  ← adjust to your local paths
# ─────────────────────────────────────────────
PATHS = {
    "pca"           : "bert_pca.pkl",
    "scaler"        : "mlp_scaler.pkl",
    "model"         : "mlp_model.pkl",
    "label_encoder" : "mlp_label_encoder.pkl",
    "session_log"   : "session_log.json",
}

BERT_MODEL = "bert-base-uncased"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_artifact(path, name, use_joblib=False):
    try:
        if use_joblib:
            obj = joblib.load(path)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        print(f"  ✅ {name}")
        return obj
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Cannot find {name} at: {path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load {name}: {e}")

def print_banner():
    print("\n" + "═" * 55)
    print("   MAGE AI TEXT DETECTION — INTERACTIVE INFERENCE")
    print("═" * 55)

def print_result(res, sample_num):
    label_icon = "🤖  AI-GENERATED" if "ai" in res["prediction"].lower() else "🧑  HUMAN-WRITTEN"
    print(f"\n  ┌───────────────────────────────────────────┐")
    print(f"  │  Sample #{sample_num:<5}  →  {label_icon:<24}│")
    print(f"  ├───────────────────────────────────────────┤")
    print(f"  │  P(AI)      : {str(res['prob_ai']):<10}                 │")
    print(f"  │  P(Human)   : {str(res['prob_human']):<10}                 │")
    print(f"  │  Confidence : {str(res['confidence']):<10}  {res['tier']:<14}│")
    print(f"  └───────────────────────────────────────────┘")

def get_confidence_tier(confidence):
    if confidence >= 0.90:
        return "Very High ✅"
    elif confidence >= 0.75:
        return "High 🟢"
    elif confidence >= 0.60:
        return "Medium 🟡"
    else:
        return "Low 🔴"

# ─────────────────────────────────────────────
# 1. LOAD ARTEFACTS
# ─────────────────────────────────────────────
print_banner()
print("\n📦 Loading artefacts …")

pca    = load_artifact(PATHS["pca"],           "PCA transformer",  use_joblib=True)
scaler = load_artifact(PATHS["scaler"],        "StandardScaler",   use_joblib=False)
mlp    = load_artifact(PATHS["model"],         "MLP model",        use_joblib=False)
le     = load_artifact(PATHS["label_encoder"], "LabelEncoder",     use_joblib=False)

# ─────────────────────────────────────────────
# 2. LOAD BERT  (done once, heavy operation)
# ─────────────────────────────────────────────
print(f"\n🔄 Loading BERT tokenizer + model ({BERT_MODEL}) …")
print(   "   This may take 30–60 seconds on first run …")

from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
bert      = BertModel.from_pretrained(BERT_MODEL)
bert.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert   = bert.to(device)
print(f"  ✅ BERT loaded on {device}")

# ─────────────────────────────────────────────
# 3. TEXT → BERT EMBEDDING
# ─────────────────────────────────────────────
def get_bert_embedding(text: str) -> np.ndarray:
    """
    Tokenize text, run through BERT, return [CLS] token embedding (768-dim).
    Handles texts longer than 512 tokens by truncating.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert(**inputs)

    # [CLS] token = first token of last hidden state
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 768)
    return cls_embedding.astype(np.float32)

# ─────────────────────────────────────────────
# 4. FULL PREDICTION PIPELINE
# ─────────────────────────────────────────────
def predict_text(text: str) -> dict:
    # Step 1 — BERT embedding (768-dim)
    embedding = get_bert_embedding(text)                  # (1, 768)

    # Step 2 — PCA (768 → 80)
    reduced   = pca.transform(embedding).astype(np.float32)  # (1, 80)

    # Step 3 — StandardScaler
    scaled    = scaler.transform(reduced)                 # (1, 80)

    # Step 4 — MLP predict
    pred_int   = mlp.predict(scaled)[0]
    pred_proba = mlp.predict_proba(scaled)[0]
    label_map  = {0: "human", 1: "ai"}               # ← add this
    pred_label = label_map[int(pred_int)]             # ← replace the old line

    prob_ai    = float(pred_proba[1])
    prob_human = float(pred_proba[0])
    confidence = float(np.max(pred_proba))

    return {
        "prediction" : str(pred_label),
        "prob_ai"    : round(prob_ai,    4),
        "prob_human" : round(prob_human, 4),
        "confidence" : round(confidence, 4),
        "tier"       : get_confidence_tier(confidence),
    }

# ─────────────────────────────────────────────
# 5. INTERACTIVE LOOP
# ─────────────────────────────────────────────
print(f"\n✅ All systems ready!")
print(f"   Paste or type any text and press Enter to classify it.")
print(f"   Type  stop / quit / exit  to end the session.\n")
print("─" * 55)

session_results = []
sample_num      = 0

while True:
    # ── get input ──
    try:
        print()
        text = input("📝 Enter text: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n⚠️  Interrupted — ending session.")
        break

    # ── exit commands ──
    if text.lower() in {"stop", "quit", "exit", "q", "bye"}:
        print("\n👋 Stopping inference session.")
        break

    # ── empty input ──
    if not text:
        print("  ⚠️  Empty input — please enter some text.")
        continue

    # ── too short ──
    if len(text.split()) < 3:
        print("  ⚠️  Text too short — please enter at least a few words.")
        continue

    # ── predict ──
    print("\n  ⏳ Processing …")
    try:
        res = predict_text(text)
    except Exception as e:
        print(f"\n  ❌ Prediction failed: {e}")
        continue

    sample_num += 1
    print_result(res, sample_num)
    session_results.append({
        "sample" : sample_num,
        "text"   : text[:300],        # truncate long texts in log
        **res
    })

# ─────────────────────────────────────────────
# 6. SESSION SUMMARY
# ─────────────────────────────────────────────
if session_results:
    total    = len(session_results)
    n_ai     = sum(1 for r in session_results if "ai"    in r["prediction"].lower())
    n_human  = sum(1 for r in session_results if "human" in r["prediction"].lower())
    avg_conf = sum(r["confidence"] for r in session_results) / total

    print(f"\n{'═'*55}  SESSION SUMMARY")
    print(f"  Samples processed  : {total}")
    print(f"  🤖 AI predicted    : {n_ai}  ({100*n_ai/total:.1f}%)")
    print(f"  🧑 Human predicted : {n_human}  ({100*n_human/total:.1f}%)")
    print(f"  Avg confidence     : {avg_conf:.4f}")

    # save log
    with open(PATHS["session_log"], "w", encoding="utf-8") as f:
        json.dump({
            "total_samples"  : total,
            "n_ai"           : n_ai,
            "n_human"        : n_human,
            "avg_confidence" : round(avg_conf, 4),
            "predictions"    : session_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Session log saved → {PATHS['session_log']}")

print("\n✅ Done.\n")