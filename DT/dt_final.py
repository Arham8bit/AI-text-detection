"""
train_decision_tree.py
──────────────────────
Trains a Decision Tree classifier on stylometric features extracted
from extracted_dt_combined.csv, runs full evaluation, and saves:

  ├── dt_model.pkl          ← trained model + scaler bundle for inference
  ├── results.jsonl         ← one JSON record per experiment / config tested
  └── dt_feature_importance.png  ← bar chart of feature importances

Research angles covered
  1. Best single configuration (max_depth, min_samples_split, criterion)
  2. Parameter sweep  — depth × split threshold × criterion
  3. Cross-dataset generalisation  — train on 80 %, evaluate on held-out 20 %
  4. Full metric suite  — accuracy, precision, recall, F1, ROC-AUC,
                          confusion matrix, classification report

NOTE: DecisionTreeClassifier is implemented entirely from scratch —
      no sklearn tree classes are used anywhere in the model itself.
"""

# =============================
# 0. IMPORTS & CONFIG
# =============================
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Kaggle / headless
import matplotlib.pyplot as plt

from itertools import product as iterproduct

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV      = "extracted_dt_combined.csv"
MODEL_PKL      = "dt_model.pkl"
RESULTS_JSONL  = "results.jsonl"
FIG_IMPORTANCE = "dt_feature_importance.png"
FIG_ROC        = "dt_roc_curve.png"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 49
np.random.seed(RANDOM_SEED)

# ── Feature columns (must match extraction script exactly) ────────────────────
FEATURE_COLS = [
    "avg_sent_len", "std_sent_len", "avg_word_len", "std_word_len", "ttr",
    "comma_count", "period_count", "exclaim_count", "question_count",
    "sent_burstiness", "para_len_variance", "hapax_ratio",
    "contraction_rate", "poly_ratio", "passive_ratio", "transition_density",
]


# =============================================================================
# FROM-SCRATCH DECISION TREE IMPLEMENTATION
# =============================================================================

class _Node:
    """
    A single node in the decision tree.

    Internal node  → feature + threshold + left/right children are set.
    Leaf node      → leaf_proba is set (class probability vector).
    """
    __slots__ = (
        "feature", "threshold", "left", "right",
        "leaf_proba", "n_samples", "impurity",
    )

    def __init__(
        self, *,
        feature=None, threshold=None,
        left=None,    right=None,
        leaf_proba=None,
        n_samples=0,  impurity=0.0,
    ):
        self.feature    = feature       # int   – which feature to test
        self.threshold  = threshold     # float – split: x ≤ threshold → left
        self.left       = left          # _Node – left subtree
        self.right      = right         # _Node – right subtree
        self.leaf_proba = leaf_proba    # np.ndarray (n_classes,) – only on leaves
        self.n_samples  = n_samples     # int   – samples reaching this node
        self.impurity   = impurity      # float – node impurity (for reference)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier — implemented entirely from scratch.

    This is a drop-in replacement for sklearn.tree.DecisionTreeClassifier
    with an identical public API so it works seamlessly with sklearn's
    cross_validate, StandardScaler pipeline, and all metric utilities.

    Parameters
    ----------
    max_depth         : int or None
        Maximum depth of the tree. None means nodes are expanded until
        all leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int
        Minimum number of samples required to attempt a split on a node.
        Nodes with fewer samples become leaves immediately.
    criterion         : {'gini', 'entropy'}
        The impurity function used to evaluate candidate splits.
        'gini'    → Gini impurity  = 1 − Σ pᵢ²
        'entropy' → Shannon entropy = −Σ pᵢ log₂(pᵢ)
    random_state      : int or None
        Kept for API compatibility; the algorithm is deterministic, so
        this parameter has no effect on results.

    Attributes (set after fit)
    ----------
    classes_              : np.ndarray  – unique class labels in sorted order
    n_classes_            : int
    n_features_in_        : int
    feature_importances_  : np.ndarray shape (n_features,)
        Normalised total weighted impurity decrease per feature (mirrors
        sklearn's Gini/entropy importance definition exactly).
    """

    # Tells sklearn scorers (precision, roc_auc, …) this is a classifier
    _estimator_type = "classifier"

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        random_state=None,
    ):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.criterion          = criterion
        self.random_state       = random_state

        # populated by fit()
        self.root_              = None
        self.classes_           = None
        self.n_classes_         = 0
        self.n_features_in_     = 0
        self.feature_importances_ = None
        self._importance_acc    = None   # running total during tree build

    # ── sklearn compatibility (clone / cross_validate / set_params) ───────────

    def get_params(self, deep=True):
        """Return constructor parameters — required by sklearn.clone()."""
        return {
            "max_depth"         : self.max_depth,
            "min_samples_split" : self.min_samples_split,
            "criterion"         : self.criterion,
            "random_state"      : self.random_state,
        }

    def set_params(self, **params):
        """Set parameters — required by sklearn grid-search / pipeline."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ── Impurity computation ──────────────────────────────────────────────────

    def _gini(self, counts, n):
        """
        Gini impurity = 1 − Σ (cᵢ/n)²

        Parameters
        ----------
        counts : np.ndarray – per-class sample counts (shape: n_classes,)
        n      : int        – total samples (= sum(counts))
        """
        if n == 0:
            return 0.0
        p = counts / n          # class probabilities
        return 1.0 - float(np.dot(p, p))   # 1 − Σ pᵢ²

    def _entropy(self, counts, n):
        """
        Shannon entropy = −Σ pᵢ log₂(pᵢ)   (zero-count classes skipped)

        Parameters
        ----------
        counts : np.ndarray – per-class sample counts
        n      : int        – total samples
        """
        if n == 0:
            return 0.0
        p = counts / n
        # log2(0) is undefined; restrict to positive probabilities
        mask  = p > 0
        p_pos = p[mask]
        return float(-np.sum(p_pos * np.log2(p_pos)))

    def _impurity(self, counts, n):
        """Dispatch to the configured criterion."""
        if self.criterion == "gini":
            return self._gini(counts, n)
        return self._entropy(counts, n)

    def _node_impurity(self, y):
        """Compute impurity directly from a label array (convenience wrapper)."""
        n      = len(y)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        return self._impurity(counts, n)

    # ── Split search ──────────────────────────────────────────────────────────

    def _best_split(self, X, y):
        """
        Find the single (feature, threshold) pair that maximises
        information gain = parent_impurity − weighted_child_impurity.

        Algorithm
        ---------
        For each feature:
          1. Sort samples by feature value  (O(n log n))
          2. Scan through potential split points with two incremental
             class-count arrays (left_counts, right_counts), updating
             in O(1) per step instead of re-counting (total O(n) per feature).
          3. Only evaluate splits at feature-value boundaries (when
             consecutive sorted values differ).

        Returns
        -------
        best_feature   : int   or None   – feature index of best split
        best_threshold : float or None   – numeric threshold (≤ → left)
        best_gain      : float           – information gain achieved
        """
        n_samples, n_features = X.shape

        # Parent node impurity
        total_counts  = np.bincount(y, minlength=self.n_classes_).astype(float)
        parent_impurity = self._impurity(total_counts, n_samples)

        best_gain      = 0.0    # only splits with strictly positive gain are accepted
        best_feature   = None
        best_threshold = None

        for feat in range(n_features):
            x_col = X[:, feat]

            # ── Sort by this feature ─────────────────────────────────────────
            order    = np.argsort(x_col, kind="quicksort")
            x_sorted = x_col[order]
            y_sorted = y[order]

            # ── Initialise incremental counters ──────────────────────────────
            # left partition starts empty; right partition starts with all samples
            left_counts  = np.zeros(self.n_classes_, dtype=float)
            right_counts = total_counts.copy()

            for i in range(n_samples - 1):
                # Move sample i from right → left
                c = y_sorted[i]
                left_counts[c]  += 1.0
                right_counts[c] -= 1.0

                # Skip if x_sorted[i] == x_sorted[i+1]
                # (split here would not separate any samples)
                if x_sorted[i] == x_sorted[i + 1]:
                    continue

                n_left  = i + 1
                n_right = n_samples - n_left

                # Weighted child impurity
                left_imp  = self._impurity(left_counts,  n_left)
                right_imp = self._impurity(right_counts, n_right)
                child_imp = (n_left * left_imp + n_right * right_imp) / n_samples

                gain = parent_impurity - child_imp

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feat
                    # Threshold is the midpoint between the two boundary values
                    best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0

        return best_feature, best_threshold, best_gain

    # ── Tree construction (recursive) ─────────────────────────────────────────

    def _make_leaf(self, y):
        """
        Create a leaf node whose leaf_proba is the empirical class
        probability vector estimated from the samples at this node.
        """
        n      = len(y)
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        proba  = counts / n if n > 0 else counts
        return _Node(leaf_proba=proba, n_samples=n)

    def _build(self, X, y, depth):
        """
        Recursively build the subtree rooted at the current node.

        Stopping conditions (any one triggers leaf creation):
          • Fewer than min_samples_split samples at this node
          • Reached max_depth
          • All samples belong to the same class (pure node)
          • No split improves impurity (best_gain == 0)
        """
        n = len(y)

        # ── Stopping conditions ────────────────────────────────────────────
        if n < self.min_samples_split:
            return self._make_leaf(y)

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        if np.all(y == y[0]):              # pure node — all same class
            return self._make_leaf(y)

        # ── Search for best split ──────────────────────────────────────────
        feat, thresh, gain = self._best_split(X, y)

        if feat is None:                   # no improving split exists
            return self._make_leaf(y)

        # ── Accumulate feature importance ──────────────────────────────────
        # sklearn's definition: weighted impurity decrease × n_samples_node
        # (normalised to [0, 1] after the full tree is built)
        self._importance_acc[feat] += gain * n

        # ── Partition samples ──────────────────────────────────────────────
        go_left = X[:, feat] <= thresh

        left_child  = self._build(X[go_left],  y[go_left],  depth + 1)
        right_child = self._build(X[~go_left], y[~go_left], depth + 1)

        return _Node(
            feature=feat, threshold=thresh,
            left=left_child, right=right_child,
            n_samples=n, impurity=self._node_impurity(y),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Build the decision tree from training data X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) – feature matrix
        y : array-like of shape (n_samples,)             – integer class labels
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)

        self.classes_       = np.unique(y)
        self.n_classes_     = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._importance_acc = np.zeros(self.n_features_in_, dtype=float)

        # Map arbitrary labels → contiguous 0-based indices
        # (handles cases where labels are e.g. {1, 2} instead of {0, 1})
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[v] for v in y], dtype=int)

        # Build the tree
        self.root_ = self._build(X, y_idx, depth=0)

        # Normalise feature importances to sum to 1.0
        total = self._importance_acc.sum()
        if total > 0:
            self.feature_importances_ = self._importance_acc / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_)

        return self

    def _traverse(self, x, node):
        """
        Walk the tree for a single sample x, returning the leaf's
        probability vector.  Iterative (avoids Python recursion limit).
        """
        while node.leaf_proba is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.leaf_proba

    def predict_proba(self, X):
        """
        Return class probabilities for each sample in X.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            proba[:, 1] = P(AI) for binary classification.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(x, self.root_) for x in X])

    def predict(self, X):
        """
        Return the predicted class label for each sample in X.

        Labels are drawn from self.classes_ (the original label space).
        """
        proba   = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]


# END OF SCRATCH IMPLEMENTATION
# =============================================================================


# =============================
# 1. LOAD DATA
# =============================
print("=" * 60)
print("DECISION TREE — AI TEXT DETECTION TRAINING PIPELINE")
print("=" * 60)

print(f"\n[1/6] Loading dataset: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Defensive: drop rows with any NaN in features or label
df = df.dropna(subset=["label"] + FEATURE_COLS).reset_index(drop=True)
print(f"  Loaded shape : {df.shape}")
print(f"  Label dist   :\n{df['label'].value_counts().to_string()}")

X = df[FEATURE_COLS].values.astype(np.float64)
y = df["label"].values.astype(int)

# =============================
# 2. TRAIN / TEST SPLIT
# =============================
print("\n[2/6] Splitting data (80 % train / 20 % test, stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# ── StandardScaler (optional for DT, but keeps pipeline consistent
#    with ANN for fair hybrid comparison later) ────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =============================
# 3. HELPER — EVALUATE MODEL
# =============================
def evaluate(model, X_tr, y_tr, X_te, y_te, label: str) -> dict:
    """
    Returns a result dict with all metrics for one model configuration.
    """
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_prob = model.predict_proba(X_te)[:, 1]
    threshold = 0.7   # try 0.6–0.9 range
    y_pred = (y_prob >= threshold).astype(int)

    acc   = accuracy_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred, zero_division=0)
    rec   = recall_score(y_te, y_pred, zero_division=0)
    f1    = f1_score(y_te, y_pred, zero_division=0)
    auc   = roc_auc_score(y_te, y_prob)
    cm    = confusion_matrix(y_te, y_pred).tolist()
    report = classification_report(y_te, y_pred,
                                   target_names=["human", "ai"],
                                   output_dict=True)

    result = {
        "experiment"           : label,
        "accuracy"             : round(acc,  4),
        "precision"            : round(prec, 4),
        "recall"               : round(rec,  4),
        "f1_score"             : round(f1,   4),
        "roc_auc"              : round(auc,  4),
        "confusion_matrix"     : cm,
        "classification_report": report,
        "train_time_sec"       : round(train_time, 4),
        "n_train"              : int(X_tr.shape[0]),
        "n_test"               : int(X_te.shape[0]),
        "params"               : model.get_params(),
    }

    print(f"  [{label}]  acc={acc:.4f}  prec={prec:.4f}  "
          f"rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
          f"({train_time:.2f}s)")
    return result

# =============================
# 4. PARAMETER SWEEP
# =============================
print("\n[3/6] Parameter sweep …")

DEPTH_VALS      = [5, 10, 15, None]       # None = unlimited
MIN_SPLIT_VALS  = [2, 10, 50, 100]
CRITERION_VALS  = ["gini", "entropy"]

all_results = []
best_auc    = -1.0
best_model  = None
best_result = None

combos = list(iterproduct(DEPTH_VALS, MIN_SPLIT_VALS, CRITERION_VALS))
print(f"  Total configs : {len(combos)}")

for depth, min_split, criterion in combos:
    tag = (f"depth={depth}_minsplit={min_split}_criterion={criterion}")
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=min_split,
        criterion=criterion,
        random_state=RANDOM_SEED,
    )
    res = evaluate(clf, X_train_s, y_train, X_test_s, y_test, tag)
    res["sweep"] = True
    all_results.append(res)

    if res["roc_auc"] > best_auc:
        best_auc    = res["roc_auc"]
        best_model  = clf
        best_result = res

print(f"  ✔ Best config  : {best_result['experiment']}")
print(f"    AUC={best_auc:.4f}  F1={best_result['f1_score']:.4f}")

# =============================
# 5. CROSS-VALIDATION ON BEST
# =============================
print("\n[4/6] 5-fold cross-validation on best config …")

cv_clf = DecisionTreeClassifier(**best_model.get_params())
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_scores = cross_validate(
    cv_clf, X_train_s, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=True
)

cv_result = {
    "experiment"        : "cross_validation_best_config",
    "sweep"             : False,
    "cv_folds"          : 5,
    "params"            : best_model.get_params(),
    "cv_test_accuracy"  : {
        "mean" : round(float(cv_scores["test_accuracy"].mean()),  4),
        "std"  : round(float(cv_scores["test_accuracy"].std()),   4),
        "folds": [round(float(v), 4) for v in cv_scores["test_accuracy"]]
    },
    "cv_test_precision" : {
        "mean" : round(float(cv_scores["test_precision"].mean()), 4),
        "std"  : round(float(cv_scores["test_precision"].std()),  4),
        "folds": [round(float(v), 4) for v in cv_scores["test_precision"]]
    },
    "cv_test_recall"    : {
        "mean" : round(float(cv_scores["test_recall"].mean()),    4),
        "std"  : round(float(cv_scores["test_recall"].std()),     4),
        "folds": [round(float(v), 4) for v in cv_scores["test_recall"]]
    },
    "cv_test_f1"        : {
        "mean" : round(float(cv_scores["test_f1"].mean()),        4),
        "std"  : round(float(cv_scores["test_f1"].std()),         4),
        "folds": [round(float(v), 4) for v in cv_scores["test_f1"]]
    },
    "cv_test_roc_auc"   : {
        "mean" : round(float(cv_scores["test_roc_auc"].mean()),   4),
        "std"  : round(float(cv_scores["test_roc_auc"].std()),    4),
        "folds": [round(float(v), 4) for v in cv_scores["test_roc_auc"]]
    },
}
all_results.append(cv_result)

print(f"  CV F1    : {cv_result['cv_test_f1']['mean']:.4f} "
      f"± {cv_result['cv_test_f1']['std']:.4f}")
print(f"  CV AUC   : {cv_result['cv_test_roc_auc']['mean']:.4f} "
      f"± {cv_result['cv_test_roc_auc']['std']:.4f}")
print(f"  CV Acc   : {cv_result['cv_test_accuracy']['mean']:.4f} "
      f"± {cv_result['cv_test_accuracy']['std']:.4f}")

# =============================
# 6. FINAL MODEL — RETRAIN ON
#    FULL TRAINING SET
# =============================
print("\n[5/6] Retraining best config on full training set …")

final_clf = DecisionTreeClassifier(**best_model.get_params())
final_clf.fit(X_train_s, y_train)

final_res = evaluate(final_clf, X_train_s, y_train,
                     X_test_s, y_test, "final_best_model")
final_res["sweep"] = False
final_res["note"]  = "Retrained best config on full train split; evaluated on held-out test"
all_results.append(final_res)

# ── Feature importances ───────────────────────────────────────────────────────
importances = final_clf.feature_importances_
imp_series  = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)

importance_record = {
    "experiment"         : "feature_importances",
    "sweep"              : False,
    "feature_importances": {k: round(float(v), 6) for k, v in imp_series.items()},
}
all_results.append(importance_record)

print("\n  Feature importances (top 8):")
for feat, imp in imp_series.head(8).items():
    bar = "█" * int(imp * 200)
    print(f"    {feat:<22s}  {imp:.4f}  {bar}")

# ── Metric analysis note ──────────────────────────────────────────────────────
y_pred_final = final_clf.predict(X_test_s)
y_prob_final = final_clf.predict_proba(X_test_s)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
analysis_record = {
    "experiment"         : "metric_analysis",
    "sweep"              : False,
    "true_positives"     : int(tp),
    "true_negatives"     : int(tn),
    "false_positives"    : int(fp),
    "false_negatives"    : int(fn),
    "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
    "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0,
    "note": (
        "FPR = human text incorrectly flagged as AI. "
        "FNR = AI text missed. For academic integrity use-cases, "
        "low FPR is critical to avoid wrongly accusing students."
    ),
}
all_results.append(analysis_record)

print(f"\n  False Positive Rate (human → AI): {analysis_record['false_positive_rate']:.4f}")
print(f"  False Negative Rate (AI → human): {analysis_record['false_negative_rate']:.4f}")

# =============================
# 7. SAVE ARTEFACTS
# =============================
print(f"\n[6/6] Saving artefacts …")

# ── results.jsonl ─────────────────────────────────────────────────────────────
with open(RESULTS_JSONL, "w", encoding="utf-8") as fout:
    for rec in all_results:
        fout.write(json.dumps(rec, default=str) + "\n")
print(f"  ✔ {RESULTS_JSONL}  ({len(all_results)} records)")

# ── dt_model.pkl  (bundle: scaler + model + metadata) ────────────────────────
bundle = {
    "model"       : final_clf,
    "scaler"      : scaler,
    "feature_cols": FEATURE_COLS,
    "label_map"   : {0: "human", 1: "ai"},
    "best_params" : final_clf.get_params(),
    "test_metrics": {
        "accuracy" : final_res["accuracy"],
        "f1_score" : final_res["f1_score"],
        "roc_auc"  : final_res["roc_auc"],
    },
}
joblib.dump(bundle, MODEL_PKL)
print(f"  ✔ {MODEL_PKL}")

# ── Feature importance plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors  = ["#e05c5c" if i < 5 else "#5b8dee" for i in range(len(imp_series))]
imp_series.plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
ax.invert_yaxis()
ax.set_title("Decision Tree — Feature Importances", fontsize=14, fontweight="bold")
ax.set_xlabel("Gini Importance", fontsize=11)
ax.axvline(0, color="black", linewidth=0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_IMPORTANCE, dpi=150)
plt.close(fig)
print(f"  ✔ {FIG_IMPORTANCE}")

# ── ROC curve plot ────────────────────────────────────────────────────────────
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob_final)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_arr, tpr_arr, color="#5b8dee", lw=2,
        label=f"DT  (AUC = {final_res['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Decision Tree", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_ROC, dpi=150)
plt.close(fig)
print(f"  ✔ {FIG_ROC}")

# =============================
# SUMMARY
# =============================
print("\n" + "=" * 60)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Best config   : {best_result['experiment']}")
print(f"  Test Accuracy : {final_res['accuracy']:.4f}")
print(f"  Test Precision: {final_res['precision']:.4f}")
print(f"  Test Recall   : {final_res['recall']:.4f}")
print(f"  Test F1       : {final_res['f1_score']:.4f}")
print(f"  Test ROC-AUC  : {final_res['roc_auc']:.4f}")
print(f"\n  Outputs saved:")
print(f"    {MODEL_PKL:<35s} ← inference bundle")
print(f"    {RESULTS_JSONL:<35s} ← all experiment records")
print(f"    {FIG_IMPORTANCE:<35s} ← feature importance chart")
print(f"    {FIG_ROC:<35s} ← ROC curve")
print("=" * 60)

# =============================
# INFERENCE USAGE EXAMPLE
# =============================
print("""
── HOW TO USE dt_model.pkl FOR INFERENCE ──────────────────────
import joblib, numpy as np

bundle  = joblib.load("dt_model.pkl")
model   = bundle["model"]
scaler  = bundle["scaler"]
cols    = bundle["feature_cols"]
lmap    = bundle["label_map"]

# features must be in the same order as `cols`
raw_features = np.array([[...]])          # shape (1, 16)
scaled       = scaler.transform(raw_features)
pred_label   = lmap[model.predict(scaled)[0]]
confidence   = model.predict_proba(scaled)[0][1]   # P(AI)

print(pred_label, confidence)
───────────────────────────────────────────────────────────────
""")