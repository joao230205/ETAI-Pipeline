"""Evaluating a model honestly: cross-validation of the whole pipeline with several metrics,
out-of-fold probabilities, the fairness audit computed on them, and a corrected paired test."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    Turns metric names into scorers. Any sklearn scorer name works ("accuracy", "recall",
    "roc_auc", ...); `fbeta_<beta>` (e.g. "fbeta_2") is F-beta, which weighs recall `beta`
    times as much as precision.
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            scorers[name] = make_scorer(fbeta_score, beta=beta, zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` on each fold's training part and scores it on that
    fold's validation part, with every scorer in `scorers`. Because `pipeline` contains the
    preprocessing too, imputation medians, encoder statistics and scaler means are re-learned
    inside every fold -- the validation fold never influences its own preprocessing.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold -- train / validation / gap for `selection_metric`
        (a large, consistent gap = overfitting), then one column per metric (validation).
      - oof_proba: out-of-fold P(positive class). Every row gets a probability from the one
        fold model that did NOT train on it. Labels are derived from it at any threshold, and
        the ranking metrics (roc_auc, average_precision) need the probabilities themselves.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores[f"test_{selection_metric}"]) + 1),
        "train": scores[f"train_{selection_metric}"],
        "validation": scores[f"test_{selection_metric}"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str = "") -> str:
    """Per-fold table + mean +/- std for every metric, as text (the caller prints / saves it)."""
    lines = [
        f"{title or 'Cross-validation'} ({len(fold_scores)} stratified folds, "
        f"selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize() + ' (' + selection_metric + ')':<30s} mean = "
                     f"{fold_scores[col].mean():{sign}.3f}   std = {fold_scores[col].std(ddof=1):.3f}")
    lines += ["", "All metrics on the validation folds:"]
    for name in metrics:
        lines.append(f"  {name:<20s} mean = {fold_scores[name].mean():.3f}   "
                     f"std = {fold_scores[name].std(ddof=1):.3f}")
    return "\n".join(lines)


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Classification report and confusion matrix on the out-of-fold predictions."""
    y_pred = (np.asarray(oof_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return (f"Classification report (out-of-fold predictions, development set, threshold={threshold}):\n"
            + classification_report(y_true, y_pred, zero_division=0)
            + "\nConfusion matrix (rows = actual 0/1, columns = predicted 0/1):\n"
            + np.array2string(cm))


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough
    to show that "accurate" and "fair" are not the same thing.

    For each group of `sensitive_attr`, the false positive rate (share of people who did NOT
    reoffend but were predicted to) for:
        - our own model (out-of-fold predictions on the development set)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk"
          prediction), on the same rows, for comparison
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (np.asarray(oof_proba) >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold, threshold={threshold})",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float):
    """
    Corrected resampled paired t-test (Nadeau & Bengio, 2003) on two models' per-fold scores
    taken on the SAME folds. Folds share training rows, so the plain paired t-test is too
    confident; the variance is inflated by (1/k + n_test/n_train). For k-fold,
    test_train_ratio = 1 / (k - 1). Returns (mean difference A - B, t statistic, two-sided p).
    """
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), 0.0 if d.mean() == 0 else np.inf, 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return d.mean(), t, p
