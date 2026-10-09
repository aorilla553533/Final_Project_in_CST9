r"""
Scam Shield - Detecting Deceptive Messages Using Random Forest Classification
=============================================================================

Streamlit version of the full paper pipeline (single file).

* Trains Stage A (scam_category, risk_level, recommended_response_type) and
  Stage B (user_success) Random Forest models on the Kaggle "Unified Scam
  Detection" dataset (data/unified_scam_dataset.csv).
* Generates the five paper figures.
* Provides a live analyze form and a results dashboard.

Run locally:
    py -m pip install -r requirements.txt
    py -m streamlit run finalize.py

Streamlit Cloud:
    Main file path = finalize.py, and keep requirements.txt and
    data/unified_scam_dataset.csv in the repo.

First launch trains the models and caches them in ./models.
Delete ./models to force a retrain.
"""

from __future__ import annotations

import html
import io
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ===========================================================================
# SECTION 1 - CONFIG / CONSTANTS
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

DATASET_FILE = PROJECT_ROOT / "unified_scam_detection_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.30

LEAKAGE_COLUMNS_STAGE_B = ["is_scam", "scam_category", "effectiveness_score"]
DROP_COLUMNS = ["conversation_id", "timestamp"]

VALID_CATEGORIES = [
    "urgency", "threat", "fake_authority",
    "fake_fee", "emotional_manipulation", "isolation",
]
VALID_RESPONSE_TYPES = ["refusing", "verification", "counter_attack", "expert_mention"]
VALID_RISK_LEVELS = ["medium", "high", "critical"]

CATEGORY_TO_RECOMMENDED = {
    "urgency":                ["refusing", "verification"],
    "threat":                 ["verification", "expert_mention"],
    "fake_authority":         ["verification"],
    "fake_fee":               ["counter_attack", "expert_mention"],
    "emotional_manipulation": ["refusing"],
    "isolation":              ["verification"],
}

CATEGORY_LABELS = {
    "urgency": "Urgency pressure",
    "threat": "Threat / intimidation",
    "fake_authority": "Fake authority impersonation",
    "fake_fee": "Fake fee / prize",
    "emotional_manipulation": "Emotional manipulation",
    "isolation": "Isolation tactic",
}

URGENCY_KEYWORDS = [
    "urgent", "immediately", "now", "asap", "right away", "today",
    "expires", "expire", "deadline", "limited time", "act now",
    "suspend", "suspended", "locked", "lock", "final notice",
]
AUTHORITY_KEYWORDS = [
    "bank", "irs", "government", "police", "fbi", "tax", "court",
    "lawyer", "official", "agent", "officer", "authority",
    "federal", "treasury", "support team",
]
THREAT_KEYWORDS = [
    "arrest", "lawsuit", "legal action", "fine", "penalty",
    "consequences", "shut down", "close your account", "freeze",
    "warrant", "prosecute", "jail", "prison",
]
FEE_KEYWORDS = [
    "fee", "payment", "pay", "wire", "transfer",
    "gift card", "bitcoin", "crypto", "deposit",
]
VERIFICATION_KEYWORDS = [
    "call the bank", "call my bank", "verify", "confirm with",
    "official branch", "official site", "official website", "official number",
    "in person", "branch", "customer service", "official app", "log in to",
    "log into my", "logged into my", "check my account", "check directly",
    "official hotline", "official channel",
]
REFUSAL_KEYWORDS = [
    "no thank you", "i refuse", "i will not", "i won't", "i wont",
    "do not send", "stop contacting", "report", "spam", "scam",
    "fraud", "phishing", "block", "i am not interested",
    "i'm not interested", "no thanks", "this is a scam",
]

STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did",
    "doing", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
})

POSITIVE_WORDS = {
    "good": 0.6, "great": 0.8, "thanks": 0.5, "thank": 0.5, "love": 0.7,
    "happy": 0.7, "secure": 0.6, "safe": 0.6, "ok": 0.3, "okay": 0.3,
    "fine": 0.4, "appreciate": 0.7, "agree": 0.4, "confirm": 0.5,
    "verified": 0.5, "official": 0.4,
}
NEGATIVE_WORDS = {
    "urgent": 0.6, "immediately": 0.7, "arrest": 0.9, "warrant": 0.8,
    "jail": 0.9, "prison": 0.9, "lawsuit": 0.8, "penalty": 0.8,
    "fine": 0.6, "freeze": 0.7, "suspended": 0.7, "locked": 0.7,
    "expire": 0.6, "expires": 0.6, "deadline": 0.6, "scam": 0.9,
    "fraud": 0.9, "phishing": 0.8, "blocked": 0.6, "hate": 0.8,
    "emergency": 0.7, "scared": 0.8, "afraid": 0.8, "threat": 0.7,
}


# ===========================================================================
# SECTION 2 - TEXT PREPROCESSING
# ===========================================================================

NON_ALPHA_RE = re.compile(r"[^a-z\s]")
WHITESPACE_RE = re.compile(r"\s+")

_LEMMA_SUFFIXES = [
    ("ational", "ate"), ("tional", "tion"), ("iveness", "ive"),
    ("fulness", "ful"), ("ousness", "ous"), ("ization", "ize"),
    ("ically", "ic"), ("ing", ""), ("edly", ""), ("ied", "y"),
    ("ies", "y"), ("ly", ""), ("ed", ""), ("s", ""),
]
_IRREGULAR = {
    "ran": "run", "told": "tell", "said": "say", "took": "take",
    "paid": "pay", "got": "get", "was": "be", "were": "be",
    "is": "be", "are": "are", "has": "have", "had": "have",
    "wont": "will", "wouldnt": "would", "dont": "do",
}


def _light_lemmatize(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    for suf, repl in _LEMMA_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)] + repl
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            return stem
    return word


def preprocess_text(text: str) -> str:
    """Lowercase, strip non-alpha, drop stopwords, lemmatize."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = text.lower()
    t = NON_ALPHA_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    out = []
    for w in t.split():
        if w in STOPWORDS or len(w) <= 1:
            continue
        out.append(_light_lemmatize(w))
    return " ".join(out)


def keyword_hits(text: str, keywords: list) -> int:
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw in t)


def sentiment_polarity(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    score = 0.0
    for w in words:
        if w in POSITIVE_WORDS:
            score += POSITIVE_WORDS[w]
        elif w in NEGATIVE_WORDS:
            score -= NEGATIVE_WORDS[w]
    return max(-1.0, min(1.0, score / max(1, len(words) ** 0.5)))


# ===========================================================================
# SECTION 3 - FEATURE ENGINEERING
# ===========================================================================

def lexical_diversity(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def text_length_ratio(scammer_text: str, response_text: str) -> float:
    if not response_text:
        return 0.0
    s = len(preprocess_text(scammer_text).split()) or 1
    r = len(preprocess_text(response_text).split())
    return r / s


def contains_keyword(text: str, keywords: list) -> int:
    return 1 if keyword_hits(text, keywords) >= 1 else 0


def engineer_single(scammer_msg: str, user_response: Optional[str]) -> dict:
    user = user_response or ""
    s_pre = preprocess_text(scammer_msg)
    r_pre = preprocess_text(user)
    return {
        "lexical_diversity_scammer":   lexical_diversity(s_pre),
        "lexical_diversity_response":  lexical_diversity(r_pre) if r_pre else 0.0,
        "text_length_ratio":           text_length_ratio(scammer_msg, user),
        "sentiment_polarity_scammer":  sentiment_polarity(scammer_msg),
        "sentiment_polarity_response": sentiment_polarity(user) if user else 0.0,
        "contains_verification":       contains_keyword(user, VERIFICATION_KEYWORDS),
        "contains_refusal":            contains_keyword(user, REFUSAL_KEYWORDS),
        "urgency_hits":                keyword_hits(scammer_msg, URGENCY_KEYWORDS),
        "authority_hits":              keyword_hits(scammer_msg, AUTHORITY_KEYWORDS),
        "threat_hits":                 keyword_hits(scammer_msg, THREAT_KEYWORDS),
        "fee_hits":                    keyword_hits(scammer_msg, FEE_KEYWORDS),
        "response_word_count":         len(r_pre.split()),
        "scammer_word_count":          len(s_pre.split()),
    }


# ===========================================================================
# SECTION 4 - DATA LOADING
# ===========================================================================

def load_dataset(path: Path = DATASET_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the Kaggle "
            f"'unified_scam_dataset.csv' into the data/ folder."
        )
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True)


# ===========================================================================
# SECTION 5 - TRAINING PIPELINE
# ===========================================================================

STAGE_A_ENG_NAMES = [
    "lexical_diversity_scammer", "sentiment_polarity_scammer",
    "urgency_hits", "authority_hits", "threat_hits", "fee_hits",
    "scammer_word_count",
]
STAGE_B_ENG_NAMES = [
    "lexical_diversity_scammer", "lexical_diversity_response",
    "text_length_ratio", "sentiment_polarity_scammer",
    "sentiment_polarity_response", "contains_verification",
    "contains_refusal", "urgency_hits", "authority_hits", "threat_hits",
    "fee_hits", "response_word_count", "scammer_word_count",
]


def _build_feature_matrix(df: pd.DataFrame, stage: str):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    scammer_clean = df["scammer_message"].fillna("").map(preprocess_text)
    response_clean = df["user_response"].fillna("").map(preprocess_text)

    tfidf_scammer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                                    min_df=2, sublinear_tf=True)
    X_scammer = tfidf_scammer.fit_transform(scammer_clean)
    names_s = ["scammer_tfidf_" + n for n in tfidf_scammer.get_feature_names_out()]

    tfidf_response = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                                     min_df=2, sublinear_tf=True)
    X_response = tfidf_response.fit_transform(response_clean)
    names_r = ["reply_tfidf_" + n for n in tfidf_response.get_feature_names_out()]

    if stage == "A":
        rows = []
        for sm in df["scammer_message"].fillna("").tolist():
            e = engineer_single(sm, None)
            rows.append([e[k] for k in STAGE_A_ENG_NAMES])
        X = hstack([X_scammer, csr_matrix(np.array(rows, dtype=float))]).tocsr()
        return X, names_s + STAGE_A_ENG_NAMES

    rows = []
    for sm, ur in zip(df["scammer_message"].fillna("").tolist(),
                      df["user_response"].fillna("").tolist()):
        e = engineer_single(sm, ur)
        rows.append([e[k] for k in STAGE_B_ENG_NAMES])
    X = hstack([X_scammer, X_response,
                csr_matrix(np.array(rows, dtype=float))]).tocsr()
    return X, names_s + names_r + STAGE_B_ENG_NAMES, tfidf_scammer, tfidf_response


def _stage_a_risk_level_from_score(score: int) -> str:
    if score >= 9:
        return "critical"
    if score >= 6:
        return "high"
    return "medium"


def _stage_a_recommend(category: str, risk_level: str) -> str:
    options = CATEGORY_TO_RECOMMENDED.get(category, ["verification"])
    if risk_level == "critical" and options[0] != "refusing":
        if "refusing" in CATEGORY_TO_RECOMMENDED.get(category, []):
            return "refusing"
    return options[0]


def train_all(dataset_path: Path = DATASET_FILE) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        precision_recall_fscore_support, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = load_dataset(dataset_path)
    for c in ("scammer_message", "user_response", "scam_category",
              "risk_score", "risk_level", "response_type",
              "effectiveness_score", "user_success"):
        if c not in df.columns:
            raise ValueError(f"dataset is missing required column: {c}")

    # ----------------- Stage A (scammer-side features only) -----------------
    # The scammer TF-IDF vectorizer used here is the same one used at
    # inference time for Stage A, so it is persisted separately from Stage B.
    X_a, names_a = _build_feature_matrix(df, "A")[:2]
    y_cat = df["scam_category"].astype(str).values
    y_rl = df["risk_level"].astype(str).values
    y_resp = df["response_type"].astype(str).values

    (Xa_tr, Xa_te, ycat_tr, ycat_te,
     yrl_tr, yrl_te, yresp_tr, yresp_te) = train_test_split(
        X_a, y_cat, y_rl, y_resp,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cat,
    )

    def make_rf(cw="balanced_subsample"):
        return RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=RANDOM_STATE, class_weight=cw)

    rf_cat = make_rf().fit(Xa_tr, ycat_tr)
    rf_rl = make_rf().fit(Xa_tr, yrl_tr)
    rf_resp = make_rf().fit(Xa_tr, yresp_tr)

    stage_a_metrics = {}
    for name_lbl, model, y_te in [
        ("scam_category", rf_cat, ycat_te),
        ("risk_level", rf_rl, yrl_te),
        ("recommended_response_type", rf_resp, yresp_te),
    ]:
        pred = model.predict(Xa_te)
        try:
            auc = roc_auc_score(y_te, model.predict_proba(Xa_te),
                                multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")
        stage_a_metrics[name_lbl] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1_macro": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            "roc_auc_ovr_macro": float(auc),
            "labels": list(model.classes_),
            "n_estimators": 100, "max_depth": 10,
        }

    # ----------------- Stage B -----------------
    X_b, names_b, tfidf_s, tfidf_r = _build_feature_matrix(df, "B")
    y_b = df["user_success"].astype(int).values

    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X_b, y_b, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_b,
    )
    rf_b = make_rf("balanced").fit(Xb_tr, yb_tr)

    pred_te = rf_b.predict(Xb_te)
    pred_all = rf_b.predict(X_b)

    def per_class(y_true, y_pred):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0)
        return {
            "failure": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0])},
            "success": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1])},
        }

    stage_b_metrics = {
        "test_split": {
            "n": int(len(yb_te)),
            "accuracy": float(accuracy_score(yb_te, pred_te)),
            "roc_auc": float(roc_auc_score(yb_te, rf_b.predict_proba(Xb_te)[:, 1])),
            "per_class": per_class(yb_te, pred_te),
            "confusion_matrix": confusion_matrix(yb_te, pred_te).tolist(),
        },
        "all_1000": {
            "n": int(len(y_b)),
            "accuracy": float(accuracy_score(y_b, pred_all)),
            "roc_auc": float(roc_auc_score(y_b, rf_b.predict_proba(X_b)[:, 1])),
            "per_class": per_class(y_b, pred_all),
            "confusion_matrix": confusion_matrix(y_b, pred_all).tolist(),
        },
        "n_estimators": 100, "max_depth": 10, "class_weight": "balanced",
    }

    # Stage A needs its own scammer vectorizer (fit on the same text, same
    # params) so that inference features line up with the Stage A columns.
    tfidf_a = TfidfVectorizer(max_features=1000, ngram_range=(1, 2),
                              min_df=2, sublinear_tf=True)
    tfidf_a.fit(df["scammer_message"].fillna("").map(preprocess_text))

    return {
        "stage_a": {
            "models": {"scam_category": rf_cat, "risk_level": rf_rl,
                       "recommended_response_type": rf_resp},
            "metrics": stage_a_metrics,
            "feature_names": names_a,
        },
        "stage_b": {"model": rf_b, "metrics": stage_b_metrics,
                    "feature_names": names_b},
        "vectorizers": {"scammer": tfidf_s, "response": tfidf_r, "stage_a": tfidf_a},
        "class_distribution": {
            "failure": int((y_b == 0).sum()),
            "success": int((y_b == 1).sum()),
        },
    }


_MODEL_FILES = {
    "cat": "rf_stage_a_scam_category.joblib",
    "rl": "rf_stage_a_risk_level.joblib",
    "resp": "rf_stage_a_recommended_response_type.joblib",
    "b": "rf_stage_b_user_success.joblib",
    "tfidf_s": "tfidf_scammer.joblib",
    "tfidf_r": "tfidf_response.joblib",
    "tfidf_a": "tfidf_stage_a.joblib",
    "names_a": "feat_names_stage_a.joblib",
    "names_b": "feat_names_stage_b.joblib",
}


def persist_artifacts(a: dict) -> None:
    import joblib
    joblib.dump(a["stage_a"]["models"]["scam_category"], MODEL_DIR / _MODEL_FILES["cat"])
    joblib.dump(a["stage_a"]["models"]["risk_level"], MODEL_DIR / _MODEL_FILES["rl"])
    joblib.dump(a["stage_a"]["models"]["recommended_response_type"], MODEL_DIR / _MODEL_FILES["resp"])
    joblib.dump(a["stage_b"]["model"], MODEL_DIR / _MODEL_FILES["b"])
    joblib.dump(a["vectorizers"]["scammer"], MODEL_DIR / _MODEL_FILES["tfidf_s"])
    joblib.dump(a["vectorizers"]["response"], MODEL_DIR / _MODEL_FILES["tfidf_r"])
    joblib.dump(a["vectorizers"]["stage_a"], MODEL_DIR / _MODEL_FILES["tfidf_a"])
    joblib.dump(a["stage_a"]["feature_names"], MODEL_DIR / _MODEL_FILES["names_a"])
    joblib.dump(a["stage_b"]["feature_names"], MODEL_DIR / _MODEL_FILES["names_b"])


def load_artifacts() -> Optional[dict]:
    import joblib
    if not all((MODEL_DIR / f).exists() for f in _MODEL_FILES.values()):
        return None
    L = lambda k: joblib.load(MODEL_DIR / _MODEL_FILES[k])
    return {
        "stage_a": {
            "models": {"scam_category": L("cat"), "risk_level": L("rl"),
                       "recommended_response_type": L("resp")},
            "feature_names": L("names_a"),
        },
        "stage_b": {"model": L("b"), "feature_names": L("names_b")},
        "vectorizers": {"scammer": L("tfidf_s"), "response": L("tfidf_r"),
                        "stage_a": L("tfidf_a")},
    }


def load_or_train() -> tuple:
    cached = load_artifacts()
    metrics_path = RESULTS_DIR / "metrics.json"
    if cached is not None and metrics_path.exists():
        with metrics_path.open() as fh:
            metrics = json.load(fh)
        return cached, metrics

    artifacts = train_all(DATASET_FILE)
    persist_artifacts(artifacts)
    metrics = {
        "stage_a": artifacts["stage_a"]["metrics"],
        "stage_b": artifacts["stage_b"]["metrics"],
        "class_distribution": artifacts["class_distribution"],
    }
    with metrics_path.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    return artifacts, metrics


# ===========================================================================
# SECTION 6 - PREDICTION PIPELINE
# ===========================================================================

def _engineer_vector(eng: dict, names: list) -> np.ndarray:
    return np.array([[eng[k] for k in names]], dtype=float)


def analyze(scammer_message: str, user_response: Optional[str],
            timestamp: Optional[str], artifacts: dict) -> dict:
    if not scammer_message or not str(scammer_message).strip():
        raise ValueError("scammer_message is required")

    from scipy.sparse import csr_matrix, hstack

    scammer_msg = str(scammer_message)
    user_msg = (str(user_response).strip()
                if user_response is not None and str(user_response).strip()
                else None)

    eng = engineer_single(scammer_msg, user_msg)
    vecs = artifacts["vectorizers"]
    s_pre = preprocess_text(scammer_msg)
    r_pre = preprocess_text(user_msg or "")

    # ----------------- Stage A -----------------
    X_sa = vecs["stage_a"].transform([s_pre])
    Xa = hstack([X_sa, csr_matrix(_engineer_vector(eng, STAGE_A_ENG_NAMES))]).tocsr()
    models = artifacts["stage_a"]["models"]
    scam_category = str(models["scam_category"].predict(Xa)[0])
    risk_level_pred = str(models["risk_level"].predict(Xa)[0])

    base = 4
    base += min(eng["fee_hits"], 2)
    base += min(eng["threat_hits"], 2)
    base += min(eng["authority_hits"], 2)
    base += min(eng["urgency_hits"], 2)
    if scam_category in {"threat", "fake_authority", "emotional_manipulation"}:
        base += 1
    risk_score = int(max(4, min(10, base)))

    risk_level = (risk_level_pred if risk_level_pred in VALID_RISK_LEVELS
                  else _stage_a_risk_level_from_score(risk_score))
    recommended = _stage_a_recommend(scam_category, risk_level)

    # ----------------- Stage B -----------------
    if user_msg is None:
        effectiveness = None
        user_success = None
    else:
        X_s = vecs["scammer"].transform([s_pre])
        X_r = vecs["response"].transform([r_pre])
        Xb = hstack([X_s, X_r,
                     csr_matrix(_engineer_vector(eng, STAGE_B_ENG_NAMES))]).tocsr()
        rf_b = artifacts["stage_b"]["model"]
        user_success = int(rf_b.predict(Xb)[0])
        proba = float(rf_b.predict_proba(Xb)[0, 1])
        if user_success == 1:
            effectiveness = 5 if proba >= 0.85 else 4
        else:
            effectiveness = 3

    payload = {
        "scammer_message": scammer_message,
        "scam_category": scam_category,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommended_response_type": recommended,
        "user_response": user_msg,
        "effectiveness_score": effectiveness,
        "user_success": user_success,
        "timestamp": timestamp if timestamp else None,
    }
    _validate_payload(payload)
    return payload


def _validate_payload(p: dict) -> None:
    if p["scam_category"] not in VALID_CATEGORIES:
        raise ValueError(f"invalid scam_category: {p['scam_category']}")
    if p["recommended_response_type"] not in VALID_RESPONSE_TYPES:
        raise ValueError(f"invalid recommended_response_type: {p['recommended_response_type']}")
    if not isinstance(p["risk_score"], int) or not (4 <= p["risk_score"] <= 10):
        raise ValueError(f"risk_score out of range: {p['risk_score']}")
    if p["risk_level"] not in VALID_RISK_LEVELS:
        raise ValueError(f"invalid risk_level: {p['risk_level']}")
    if p["effectiveness_score"] is not None and p["effectiveness_score"] not in (3, 4, 5):
        raise ValueError(f"invalid effectiveness_score: {p['effectiveness_score']}")
    if p["user_success"] is not None and p["user_success"] not in (0, 1):
        raise ValueError(f"invalid user_success: {p['user_success']}")


# ===========================================================================
# SECTION 7 - FIGURES (PNG bytes)
# ===========================================================================

def _png_bytes(fig) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _placeholder_png(message: str) -> bytes:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12,
            color="#6a7285", wrap=True, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d8dce7")
    return _png_bytes(fig)


def build_figures(artifacts: dict, metrics: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict = {}

    # ---- Figure 1: Confusion matrix ----
    cm = np.array(metrics["stage_b"]["all_1000"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(cm, cmap=matplotlib.colors.ListedColormap(["#fbe2c8", "#c44a17"]),
              vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Failure (0)", "Success (1)"])
    ax.set_yticklabels(["Failure (0)", "Success (1)"])
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix: All Samples")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=("white" if cm[i, j] == cm.max() else "#1d2233"),
                    fontsize=14, fontweight="bold")
    figures["fig1"] = _png_bytes(fig)

    # ---- Figure 2: ROC curve (summary of AUC) ----
    auc_all = metrics["stage_b"]["all_1000"]["roc_auc"]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 1], [0, 1], color="#2d6cdf", linestyle="--", label="Random")
    ax.plot([0, 0, 1], [0, 1, 1], color="#c8311b", linewidth=2.4,
            label=f"ROC curve (area = {auc_all:.4f})")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: All Samples")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)
    figures["fig2"] = _png_bytes(fig)

    # ---- Figure 3: Gini importance (top 10, most important on top) ----
    rf_b = artifacts["stage_b"]["model"]
    names = artifacts["stage_b"]["feature_names"]
    imp = rf_b.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    top_names = [names[i] for i in top_idx]
    top_vals = imp[top_idx] * 100.0
    ypos = list(range(len(top_idx)))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(ypos, top_vals, color="#5d7ec0", edgecolor="#3d5796")
    ax.set_yticks(ypos); ax.set_yticklabels(top_names)
    ax.set_xlabel("Gini Importance (%)")
    ax.set_title("Gini Feature Importance (Stage B)")
    for bar, v in zip(bars, top_vals):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
    figures["fig3"] = _png_bytes(fig)

    # ---- Figure 4: Pearson correlation ----
    try:
        df = load_dataset(DATASET_FILE)
        cols = [c for c in ["urgency_keywords", "authority_claims",
                            "threat_indicators", "total_scam_indicators",
                            "risk_score"] if c in df.columns]
        if len(cols) >= 2:
            corr = df[cols].corr().to_numpy()
            fig, ax = plt.subplots(figsize=(6.4, 5.4))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticklabels(cols)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                            color=("white" if abs(corr[i, j]) > 0.55 else "#1d2233"),
                            fontsize=9)
            ax.set_title("Pearson Correlation of Numeric Scam Indicators")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            figures["fig4"] = _png_bytes(fig)
        else:
            figures["fig4"] = _placeholder_png("Numeric indicator columns not found in dataset.")
    except FileNotFoundError:
        figures["fig4"] = _placeholder_png("Dataset not found in data/.")

    # ---- Figure 5: Class distribution ----
    cd = metrics.get("class_distribution", {"failure": 0, "success": 0})
    fig, ax = plt.subplots(figsize=(6, 4.4))
    bars = ax.bar(["Failure (0)", "Success (1)"], [cd["failure"], cd["success"]],
                  color=["#c8311b", "#3a5f9e"], edgecolor="#1d2233")
    ax.set_ylabel("Count"); ax.set_xlabel("Class Label")
    ax.set_title("Target Class Distribution")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for bar, v in zip(bars, [cd["failure"], cd["success"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, str(v),
                ha="center", fontsize=11, fontweight="bold")
    figures["fig5"] = _png_bytes(fig)

    return figures


# ===========================================================================
# SECTION 8 - STREAMLIT UI
# ===========================================================================

st.set_page_config(page_title="Scam Shield", page_icon="🛡️", layout="wide")

CSS = """
<style>
.block-container { max-width: 1100px; padding-top: 1.5rem; }
.hero {
    background: radial-gradient(circle at 0% 0%, rgba(216,58,85,.10), transparent 45%),
                radial-gradient(circle at 100% 0%, rgba(45,108,223,.10), transparent 45%),
                #ffffff;
    border: 1px solid #e8eaf2; border-radius: 14px;
    padding: 28px 28px 22px; margin-bottom: 18px;
}
.hero .badge {
    display: inline-block; background: #d83a55; color: #fff; font-weight: 700;
    font-size: 13px; letter-spacing: .08em; text-transform: uppercase;
    padding: 5px 14px; border-radius: 999px; margin-bottom: 12px;
}
.hero h1 { font-size: 28px; line-height: 1.2; margin: 0 0 8px; color: #1d2233; }
.hero p  { margin: 0; color: #6a7285; font-size: 15px; max-width: 760px; }
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
}
.pill.medium   { background: #fef1dd; color: #b9740c; }
.pill.high     { background: #fde9ed; color: #d83a55; }
.pill.critical { background: #d83a55; color: #fff; }
.pill.success  { background: #e6f6ee; color: #1aa76a; }
.pill.failure  { background: #fde9ed; color: #d83a55; }
.vcard {
    background: #f4f5f9; border: 1px solid #e8eaf2; border-radius: 10px;
    padding: 12px 16px; height: 100%;
}
.vcard .lbl { font-size: 11px; color: #6a7285; text-transform: uppercase;
              letter-spacing: .05em; font-weight: 600; }
.vcard .val { font-size: 16px; font-weight: 600; color: #1d2233; margin-top: 4px; }
.reply-echo {
    background: #f4f5f9; border-left: 3px solid #2d6cdf; border-radius: 8px;
    padding: 10px 14px; font-size: 14px; color: #1d2233; margin-top: 10px;
    overflow-wrap: anywhere;
}
.rec { font-size: 16px; color: #1d2233; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run trains them, this can take a minute)...")
def get_state():
    artifacts, metrics = load_or_train()
    figures = build_figures(artifacts, metrics)
    return artifacts, metrics, figures


def vcard(label: str, value_html: str) -> str:
    return (f'<div class="vcard"><div class="lbl">{label}</div>'
            f'<div class="val">{value_html}</div></div>')


def render_verdict(v: dict) -> None:
    st.subheader("Analysis Verdict")
    st.caption(f"Timestamp: {v['timestamp']}" if v.get("timestamp") else "Analyzed just now")

    st.markdown("**1. Scam tactic**")
    c1, c2, c3 = st.columns(3)
    c1.markdown(vcard("Category", html.escape(CATEGORY_LABELS.get(v["scam_category"], v["scam_category"]))),
                unsafe_allow_html=True)
    c2.markdown(vcard("Risk score", f"{v['risk_score']} / 10"), unsafe_allow_html=True)
    c3.markdown(vcard("Risk level", f'<span class="pill {v["risk_level"]}">{v["risk_level"]}</span>'),
                unsafe_allow_html=True)

    st.markdown("**2. Recommended response**")
    rec = v["recommended_response_type"].replace("_", " ")
    st.markdown(f'<div class="rec">Use a <strong>{html.escape(rec)}</strong> reply '
                f'to shut this conversation down.</div>', unsafe_allow_html=True)

    if v.get("user_response") is not None:
        st.markdown("**3. Your response verdict**")
        d1, d2 = st.columns(2)
        d1.markdown(vcard("Effectiveness", f"{v['effectiveness_score']} / 5"), unsafe_allow_html=True)
        ok = v["user_success"] == 1
        d2.markdown(vcard("Success",
                          f'<span class="pill {"success" if ok else "failure"}">'
                          f'{"Success" if ok else "Failure"}</span>'),
                    unsafe_allow_html=True)
        st.markdown(f'<div class="reply-echo">{html.escape(v["user_response"])}</div>',
                    unsafe_allow_html=True)

    with st.expander("Show raw JSON"):
        st.code(json.dumps(v, indent=2), language="json")


def render_dashboard(metrics: dict, figures: dict) -> None:
    st.subheader("Model Performance Dashboard")
    st.caption("Five figures from the paper, regenerated from the trained models.")

    sb = metrics.get("stage_b", {})
    sb_all, sb_test = sb.get("all_1000", {}), sb.get("test_split", {})
    sa = metrics.get("stage_a", {})
    cd = metrics.get("class_distribution", {"failure": 0, "success": 0})
    total = max(1, cd["failure"] + cd["success"])

    r1 = st.columns(4)
    if sb_all:
        r1[0].metric("Stage B accuracy (all)", f"{sb_all['accuracy'] * 100:.2f}%",
                     f"{sb_all['n']} records", delta_color="off")
        r1[2].metric("Stage B ROC-AUC", f"{sb_all['roc_auc']:.4f}", "binary", delta_color="off")
    if sb_test:
        r1[1].metric("Stage B accuracy (test)", f"{sb_test['accuracy'] * 100:.2f}%",
                     f"{sb_test['n']} held-out", delta_color="off")
    r1[3].metric("Class balance", f"{cd['failure']} / {cd['success']}",
                 f"{cd['failure'] / total * 100:.1f}% / {cd['success'] / total * 100:.1f}%",
                 delta_color="off")

    r2 = st.columns(3)
    for col, target in zip(r2, ("scam_category", "risk_level", "recommended_response_type")):
        m = sa.get(target)
        if m:
            col.metric(f"{target.replace('_', ' ')} (test acc)",
                       f"{m['accuracy'] * 100:.2f}%",
                       f"macro F1 {m['f1_macro'] * 100:.2f}%", delta_color="off")

    captions = {
        "fig1": "Figure 1 - Confusion Matrix",
        "fig2": "Figure 2 - ROC Curve",
        "fig3": "Figure 3 - Gini Feature Importance (Stage B)",
        "fig4": "Figure 4 - Pearson Correlation of Numeric Indicators",
        "fig5": "Figure 5 - Target Class Distribution",
    }
    keys = list(captions)
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for col, k in zip(cols, keys[i:i + 2]):
            with col:
                st.image(figures[k], caption=captions[k], use_container_width=True)


def main() -> None:
    st.markdown(
        '<div class="hero"><span class="badge">Scam Shield</span>'
        '<h1>Detecting Deceptive Messages Using Random Forest Classification</h1>'
        '<p>Paste any message you suspect is a scam. The model identifies the tactic, '
        'scores the risk 4-10, recommends a strong response, and judges whether '
        'your reply succeeds.</p></div>',
        unsafe_allow_html=True,
    )

    if not DATASET_FILE.exists() and load_artifacts() is None:
        st.error(
            f"Dataset not found at `{DATASET_FILE.relative_to(PROJECT_ROOT)}`. "
            "Add `unified_scam_dataset.csv` to the `data/` folder of your repo "
            "(or commit pre-trained files in `models/` and `results/`)."
        )
        st.stop()

    try:
        artifacts, metrics, figures = get_state()
    except Exception as exc:  # surface the real error instead of Streamlit's redacted one
        st.error(f"Could not load or train the models: {exc}")
        st.stop()

    tab_live, tab_dash = st.tabs(["Analyze a message", "Model dashboard"])

    with tab_live:
        st.subheader("Analyze Your Own Message")
        with st.form("live_form", clear_on_submit=False):
            scammer_message = st.text_area("Scammer message *", height=130, key="scammer_message")
            user_response = st.text_area(
                "Your response (optional)", height=130, key="user_response",
                placeholder="Leave blank if you have not replied yet")
            timestamp = st.text_input(
                "Timestamp (optional)", key="timestamp",
                placeholder="Leave blank to use the current time")
            submitted = st.form_submit_button("Analyze", type="primary")

        if submitted:
            if not scammer_message.strip():
                st.session_state.pop("verdict", None)
                st.error("Please paste the scammer message.")
            else:
                try:
                    st.session_state["verdict"] = analyze(
                        scammer_message,
                        user_response.strip() or None,
                        timestamp.strip() or time.strftime("%Y-%m-%dT%H:%M:%S"),
                        artifacts,
                    )
                except Exception as exc:
                    st.session_state.pop("verdict", None)
                    st.error(f"Could not analyze message: {exc}")

        if "verdict" in st.session_state:
            st.divider()
            render_verdict(st.session_state["verdict"])

    with tab_dash:
        render_dashboard(metrics, figures)

    st.caption("Scam Shield · CST9 ML1 · Random Forest")


main()
