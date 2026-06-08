import numpy as np
import pandas as pd
import json
import pickle
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, log_loss
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# HELPER: safe float formatting
# ─────────────────────────────────────────────
def safe_float(val, digits=6):
    """Return rounded float or None — never crashes on NoneType."""
    if val is None:
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None

def fmt(val, digits=4):
    """Human-readable string, graceful on None."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("MAGE AI TEXT DETECTION — MLP TRAINER")
print("=" * 60)

df = pd.read_csv("/kaggle/input/datasets/noxiousberlin/bert-extract/extracted_bert_combined (2).csv")
print(f"\n✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. IDENTIFY FEATURE COLS & LABEL COL
# ─────────────────────────────────────────────
label_col   = "label"   # ← change if your label column has a different name
feature_cols = [c for c in df.columns if c != label_col]

assert len(feature_cols) == 80, (
    f"Expected 80 BERT-PCA features, found {len(feature_cols)}. "
    "Check your label column name."
)

X     = df[feature_cols].values.astype(np.float32)
y_raw = df[label_col].values

le = LabelEncoder()
y  = le.fit_transform(y_raw)
print(f"Classes detected : {le.classes_}  →  {list(enumerate(le.classes_))}")
print(f"Class distribution:\n{pd.Series(y).value_counts().to_string()}\n")

# ─────────────────────────────────────────────
# 3. TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ─────────────────────────────────────────────
X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val,  y_train, y_val  = train_test_split(
    X_tmp, y_tmp, test_size=0.1765, random_state=42, stratify=y_tmp)

print(f"Split  →  train: {len(X_train):,}  |  val: {len(X_val):,}  |  test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. FEATURE SCALING
# ─────────────────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. BUILD & TRAIN MLP
# ─────────────────────────────────────────────
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    
    batch_size=512,
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-4,
    random_state=42,
    verbose=True
)

print("\n Training MLP …")
t0 = time.time()
mlp.fit(X_train_s, y_train)
train_time = time.time() - t0

# ── safe extraction of post-fit attributes ──
n_iter       = getattr(mlp, 'n_iter_',           None)
best_loss    = getattr(mlp, 'best_loss_',         None)
loss_curve   = getattr(mlp, 'loss_curve_',        None) or []
val_scores   = getattr(mlp, 'validation_scores_', None) or []
out_layer    = getattr(mlp, 'n_outputs_',         None)

loss_curve_clean = [safe_float(v, 8) for v in loss_curve]
val_curve_clean  = [safe_float(v, 8) for v in val_scores]

print(f"\n⏱  Training finished in {train_time:.1f}s")
print(f"   Epochs run    : {n_iter if n_iter is not None else 'N/A'}")
print(f"   Best val loss : {fmt(best_loss, 6)}")

# ─────────────────────────────────────────────
# 6. EVALUATE  (train / val / test)
# ─────────────────────────────────────────────
def evaluate(model, X_s, y_true, split_name):
    y_pred  = model.predict(X_s)
    y_proba = model.predict_proba(X_s)[:, 1]
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    def _s(fn, **kw):
        try:   return safe_float(fn(y_true, y_pred, **kw))
        except Exception: return None

    def _sp(fn, **kw):
        try:   return safe_float(fn(y_true, y_proba, **kw))
        except Exception: return None

    stats = {
        "split"           : split_name,
        "n_samples"       : int(len(y_true)),
        "accuracy"        : _s(accuracy_score),
        "f1_weighted"     : _s(f1_score,       average='weighted'),
        "f1_macro"        : _s(f1_score,        average='macro'),
        "precision_w"     : _s(precision_score, average='weighted'),
        "recall_w"        : _s(recall_score,    average='weighted'),
        "roc_auc"         : _sp(roc_auc_score),
        "mcc"             : _s(matthews_corrcoef),
        "log_loss"        : _sp(log_loss),
        "confusion_matrix": cm.tolist(),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

    print(f"\n{'─'*45}  {split_name.upper()}")
    print(f"  Accuracy   : {fmt(stats['accuracy'])}")
    print(f"  F1 (w)     : {fmt(stats['f1_weighted'])}   F1 (macro) : {fmt(stats['f1_macro'])}")
    print(f"  Precision  : {fmt(stats['precision_w'])}   Recall     : {fmt(stats['recall_w'])}")
    print(f"  ROC-AUC    : {fmt(stats['roc_auc'])}   MCC        : {fmt(stats['mcc'])}")
    print(f"  Log-Loss   : {fmt(stats['log_loss'])}")
    print(f"  Confusion Matrix  (TN={tn}  FP={fp}  FN={fn}  TP={tp}):")
    print(f"    {cm}")
    print(classification_report(y_true, y_pred,
          target_names=[str(c) for c in le.classes_]))
    return stats

train_stats = evaluate(mlp, X_train_s, y_train, "train")
val_stats   = evaluate(mlp, X_val_s,   y_val,   "validation")
test_stats  = evaluate(mlp, X_test_s,  y_test,  "test")

# ─────────────────────────────────────────────
# 7. 5-FOLD CROSS-VALIDATION  (train + val pool)
# ─────────────────────────────────────────────
print("\n⏳ Running 5-fold stratified CV on train+val set …")

cv_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', solver='adam', alpha=1e-4,
        batch_size=512, learning_rate_init=1e-3,
        max_iter=80, early_stopping=True,
        n_iter_no_change=10, random_state=42
    ))
])

skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_dev = np.vstack([X_train, X_val])
y_dev = np.concatenate([y_train, y_val])

cv_acc, cv_f1, cv_auc = [], [], []
for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_dev, y_dev), 1):
    cv_pipe.fit(X_dev[tr_idx], y_dev[tr_idx])
    yp  = cv_pipe.predict(X_dev[vl_idx])
    ypr = cv_pipe.predict_proba(X_dev[vl_idx])[:, 1]
    acc = safe_float(accuracy_score(y_dev[vl_idx], yp))
    f1  = safe_float(f1_score(y_dev[vl_idx], yp, average='weighted'))
    auc = safe_float(roc_auc_score(y_dev[vl_idx], ypr))
    cv_acc.append(acc); cv_f1.append(f1); cv_auc.append(auc)
    print(f"  Fold {fold}: acc={fmt(acc)}  f1={fmt(f1)}  auc={fmt(auc)}")

cv_stats = {
    "cv_folds"    : 5,
    "cv_acc_mean" : safe_float(np.mean(cv_acc)),
    "cv_acc_std"  : safe_float(np.std(cv_acc)),
    "cv_f1_mean"  : safe_float(np.mean(cv_f1)),
    "cv_f1_std"   : safe_float(np.std(cv_f1)),
    "cv_auc_mean" : safe_float(np.mean(cv_auc)),
    "cv_auc_std"  : safe_float(np.std(cv_auc)),
}
print(f"\n  CV Summary → "
      f"acc: {fmt(cv_stats['cv_acc_mean'])}±{fmt(cv_stats['cv_acc_std'])} | "
      f"f1: {fmt(cv_stats['cv_f1_mean'])}±{fmt(cv_stats['cv_f1_std'])} | "
      f"auc: {fmt(cv_stats['cv_auc_mean'])}±{fmt(cv_stats['cv_auc_std'])}")

# ─────────────────────────────────────────────
# 8. BUNDLE ALL STATS → JSON
# ─────────────────────────────────────────────
results = {
    "model_info": {
        "architecture"          : "MLP (256→128→64)",
        "activation"            : "relu",
        "solver"                : "adam",
        "alpha_l2"              : 1e-4,
        "batch_size"            : 512,
        "learning_rate_init"    : 1e-3,
        "max_iter"              : 100,
        "early_stopping"        : True,
        "n_iter_no_change"      : 10,
        "n_features"            : 80,
        "n_classes"             : int(len(le.classes_)),
        "classes"               : list(le.classes_.tolist()),
        "epochs_run"            : int(n_iter) if n_iter is not None else None,
        "best_val_loss"         : safe_float(best_loss, 8),
        "training_time_seconds" : round(train_time, 2),
    },
    "dataset_info": {
        "total_samples" : int(len(y)),
        "train_samples" : int(len(y_train)),
        "val_samples"   : int(len(y_val)),
        "test_samples"  : int(len(y_test)),
    },
    "train_metrics"      : train_stats,
    "validation_metrics" : val_stats,
    "test_metrics"       : test_stats,
    "cross_validation"   : cv_stats,
    "loss_curve"         : loss_curve_clean,
    "val_score_curve"    : val_curve_clean,
}

stats_path = "/kaggle/working/mlp_training_stats.json"
with open(stats_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Stats saved → {stats_path}")

# ─────────────────────────────────────────────
# 9. SAVE ARTEFACTS FOR INFERENCE
# ─────────────────────────────────────────────
model_path  = "/kaggle/working/mlp_model.pkl"
scaler_path = "/kaggle/working/mlp_scaler.pkl"
le_path     = "/kaggle/working/mlp_label_encoder.pkl"
meta_path   = "/kaggle/working/mlp_inference_meta.json"

with open(model_path,  "wb") as f: pickle.dump(mlp,    f)
with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
with open(le_path,     "wb") as f: pickle.dump(le,     f)

meta = {
    "feature_cols"   : feature_cols,
    "label_col"      : label_col,
    "label_mapping"  : {int(i): str(c) for i, c in enumerate(le.classes_)},
    "pipeline_order" : ["bert_pca.pkl  → reduce raw BERT to 80 dims",
                        "mlp_scaler.pkl → StandardScaler.transform()",
                        "mlp_model.pkl  → MLPClassifier.predict()"],
    "files": {
        "pca"           : "bert_pca.pkl",
        "scaler"        : "mlp_scaler.pkl",
        "model"         : "mlp_model.pkl",
        "label_encoder" : "mlp_label_encoder.pkl",
    }
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"💾 Model saved         → {model_path}")
print(f"💾 Scaler saved        → {scaler_path}")
print(f"💾 Label encoder saved → {le_path}")
print(f"💾 Inference meta      → {meta_path}")
print("\n✅ ALL DONE!")

# ─────────────────────────────────────────────
# 10. INFERENCE SANITY CHECK
# ─────────────────────────────────────────────
print("\n── Inference sanity check (first 5 test rows) ──")
sample    = X_test_s[:5]
pred_int  = mlp.predict(sample)
pred_lbl  = le.inverse_transform(pred_int)
pred_prob = mlp.predict_proba(sample)[:, 1]
true_lbl  = le.inverse_transform(y_test[:5])

for tl, pl, pp in zip(true_lbl, pred_lbl, pred_prob):
    match = "✓" if tl == pl else "✗"
    print(f"  [{match}] true={tl:<6}  pred={pl:<6}  P(AI)={pp:.4f}")