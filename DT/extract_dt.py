"""
extract_dt_features.py  (MAGE edition)
──────────────────────────────────────
Loads the MAGE dataset (Hugging Face: yaful/MAGE), applies the same
filtering / balancing logic as the BERT pipeline, then extracts an
expanded set of stylometric features for the Decision Tree model.

Features extracted (16 total):
  avg_sent_len, std_sent_len, avg_word_len, std_word_len, ttr,
  comma_count, period_count, exclaim_count, question_count,
  sent_burstiness, para_len_variance, hapax_ratio,
  contraction_rate, poly_ratio, passive_ratio, transition_density

Label convention:
  0 → human-written
  1 → machine-generated

Output:
  extracted_dt_combined.csv
"""

# =============================
# 0. CONFIG
# =============================
MAX_PER_CLASS = 85000
MIN_WORDS     = 100
RANDOM_SEED   = 49

import re
import numpy as np
import pandas as pd
import nltk
from datasets import load_dataset, concatenate_datasets

nltk.download("stopwords", quiet=True)

# ─── Word lists ───────────────────────────────────────────────────────────────

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

# Common transition / discourse connectors
TRANSITIONS = {
    "however","therefore","moreover","furthermore","consequently","nevertheless",
    "nonetheless","additionally","subsequently","meanwhile","accordingly",
    "alternatively","conversely","similarly","likewise","thus","hence","besides",
    "otherwise","instead","indeed","specifically","notably","importantly",
    "although","whereas","since","because","unless","until","despite",
    "provided","given","considering"
}

# Passive voice auxiliary patterns  (be-verb + past-participle heuristic)
PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE
)

# Polysyllabic words: count syllables via vowel-cluster heuristic (≥3 syllables)
VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)

def syllable_count(word: str) -> int:
    word = word.rstrip("e")          # drop silent trailing 'e'
    clusters = VOWEL_RE.findall(word)
    return max(1, len(clusters))

# =============================
# 1. LOAD MAGE DATASET
# =============================
print("Downloading MAGE dataset from Hugging Face (yaful/MAGE) …")
ds_all      = load_dataset("yaful/MAGE")
all_splits  = [ds_all[split] for split in ds_all.keys()]
ds_combined = concatenate_datasets(all_splits)
print(f"MAGE total rows (all splits combined): {len(ds_combined)}")

df_mage = ds_combined.to_pandas()[["text", "label"]]
df_mage = df_mage.dropna(subset=["text"])
df_mage["text"]  = df_mage["text"].astype(str)
df_mage["label"] = df_mage["label"].astype(int)

print(f"After NA drop           : {df_mage.shape}")
print(f"Raw label distribution  :\n{df_mage['label'].value_counts()}\n")

# ── Minimum word-count filter ─────────────────────────────────────────────────
df_mage = df_mage[df_mage["text"].apply(lambda t: len(t.split()) >= MIN_WORDS)]
df_mage = df_mage.reset_index(drop=True)
print(f"After length filter (≥{MIN_WORDS} words): {df_mage.shape}")

# ── Balance classes ───────────────────────────────────────────────────────────
human_df = df_mage[df_mage["label"] == 0]
ai_df    = df_mage[df_mage["label"] == 1]

if len(human_df) < MAX_PER_CLASS:
    raise ValueError(f"Not enough human rows: need {MAX_PER_CLASS}, got {len(human_df)}")
if len(ai_df) < MAX_PER_CLASS:
    raise ValueError(f"Not enough AI rows: need {MAX_PER_CLASS}, got {len(ai_df)}")

human_df = human_df.sample(n=MAX_PER_CLASS, random_state=RANDOM_SEED)
ai_df    = ai_df.sample(n=MAX_PER_CLASS,    random_state=RANDOM_SEED)

df = pd.concat([human_df, ai_df]).reset_index(drop=True)
print(f"\nAfter balancing : {df.shape}  ({MAX_PER_CLASS} human + {MAX_PER_CLASS} AI)")

# =============================
# 2. SHUFFLE
# =============================
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Shuffled dataset: {df.shape}")
print(f"Labels          :\n{df['label'].value_counts()}\n")

# =============================
# 3. FEATURE EXTRACTION
# =============================
def extract_stylometric_features(text: str) -> list:
    text = str(text)
    words     = text.split()
    n_words   = len(words)
    if n_words == 0:
        return [0.0] * 16

    # ── Sentence splitting ────────────────────────────────────────────────────
    sentences    = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_lengths = [len(s.split()) for s in sentences]
    n_sents      = len(sent_lengths)

    avg_sent_len = np.mean(sent_lengths) if n_sents else 0.0
    std_sent_len = np.std(sent_lengths)  if n_sents else 0.0

    # ── Word lengths ──────────────────────────────────────────────────────────
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths)
    std_word_len = np.std(word_lengths)

    # ── Type-Token Ratio ──────────────────────────────────────────────────────
    ttr = len(set(w.lower() for w in words)) / n_words

    # ── Punctuation counts ────────────────────────────────────────────────────
    comma_count   = text.count(",")
    period_count  = text.count(".")
    exclaim_count = text.count("!")
    question_count = text.count("?")

    # ── Sentence Burstiness ───────────────────────────────────────────────────
    # Coefficient of variation of sentence lengths.
    # High burstiness → very irregular sentence rhythm (more human-like).
    if n_sents > 1 and avg_sent_len > 0:
        sent_burstiness = std_sent_len / avg_sent_len
    else:
        sent_burstiness = 0.0

    # ── Paragraph Length Variance ─────────────────────────────────────────────
    # Split on blank lines; variance of paragraph word counts.
    paragraphs   = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    para_lengths = [len(p.split()) for p in paragraphs]
    para_len_variance = np.var(para_lengths) if len(para_lengths) > 1 else 0.0

    # ── Hapax Legomena Ratio ──────────────────────────────────────────────────
    # Words appearing exactly once / total words.
    from collections import Counter
    freq = Counter(w.lower() for w in words)
    hapax_ratio = sum(1 for v in freq.values() if v == 1) / n_words

    # ── Contraction Rate ──────────────────────────────────────────────────────
    # Contractions per word. Humans use these more in informal writing.
    contraction_rate = sum(
        1 for w in words if w.lower().replace("'", "'") in CONTRACTIONS
    ) / n_words

    # ── Polysyllabic (Complex Vocab) Ratio ───────────────────────────────────
    # Words with ≥3 syllables / total words.
    poly_ratio = sum(1 for w in words if syllable_count(w) >= 3) / n_words

    # ── Passive Voice Ratio ───────────────────────────────────────────────────
    # Passive constructions per sentence (heuristic: be-verb + past participle).
    passive_count = len(PASSIVE_RE.findall(text))
    passive_ratio = passive_count / n_sents if n_sents else 0.0

    # ── Transition / Conjunction Density ─────────────────────────────────────
    # Discourse connectors per word. AI text overuses these.
    transition_density = sum(
        1 for w in words if w.lower() in TRANSITIONS
    ) / n_words

    return [
        avg_sent_len,
        std_sent_len,
        avg_word_len,
        std_word_len,
        ttr,
        comma_count,
        period_count,
        exclaim_count,
        question_count,
        sent_burstiness,
        para_len_variance,
        hapax_ratio,
        contraction_rate,
        poly_ratio,
        passive_ratio,
        transition_density,
    ]


STYLO_COLUMNS = [
    "avg_sent_len",
    "std_sent_len",
    "avg_word_len",
    "std_word_len",
    "ttr",
    "comma_count",
    "period_count",
    "exclaim_count",
    "question_count",
    "sent_burstiness",
    "para_len_variance",
    "hapax_ratio",
    "contraction_rate",
    "poly_ratio",
    "passive_ratio",
    "transition_density",
]

print("Extracting stylometric features …")
features_list = df["text"].apply(extract_stylometric_features)
stylo_df      = pd.DataFrame(features_list.tolist(), columns=STYLO_COLUMNS)

final_df = pd.concat([df[["label"]].reset_index(drop=True), stylo_df], axis=1)

# =============================
# 4. SAVE
# =============================
output_path = "extracted_dt_combined.csv"
final_df.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}   shape={final_df.shape}")

# =============================
# 5. SANITY CHECK
# =============================
print("\n--- Feature means per class ---")
for col in STYLO_COLUMNS:
    h = final_df[final_df["label"] == 0][col].mean()
    a = final_df[final_df["label"] == 1][col].mean()
    print(f"  {col:<22s}  human={h:.4f}   ai={a:.4f}")

print("\nDone.")