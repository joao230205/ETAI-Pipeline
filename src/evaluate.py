"""Evaluation -- single train/test split, no cross-validation (yet)."""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate(y_train, y_train_pred, y_test, y_pred) -> str:
    """
    Prints -- and returns as text, so it can also be saved to disk -- train accuracy and test accuracy side by side, plus the usual classification report on the test set.
    """
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    gap = train_accuracy - test_accuracy

    lines = [
        f"Train accuracy: {train_accuracy:.3f}",
        f"Test accuracy:  {test_accuracy:.3f}",
        f"Gap (train - test): {gap:+.3f}",
    ]
    lines.append("")
    lines.append("Classification report (test set):")
    lines.append(classification_report(y_test, y_pred))

    text = "\n".join(lines)
    print(text)
    return text


def compare_before_after(runs: list) -> str:
    """
    Week 3: summarizes accuracy and fairness before (raw/naive
    preprocessing) vs after (clean_dataset) cleaning, for every model that
    was run on both. `runs` is a list of dicts, one per (dataset, model)
    combination, each with at minimum:
        dataset ("raw" or "cleaned"), model, train_accuracy, test_accuracy,
        n_rows, n_features, fpr_gap (max - min false positive rate across
        race groups, our model, on the test set)
    """
    lines = ["Before (raw) vs after (cleaned) -- summary", "=" * 60, ""]

    by_model = {}
    for r in runs:
        by_model.setdefault(r["model"], {})[r["dataset"]] = r

    header = f"{'Model':<22s} {'Dataset':<10s} {'Rows':>6s} {'Feats':>6s} {'Train acc':>10s} {'Test acc':>9s} {'FPR gap':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for model_name, variants in by_model.items():
        for dataset in ("raw", "cleaned"):
            r = variants.get(dataset)
            if r is None:
                continue
            lines.append(
                f"{model_name:<22s} {dataset:<10s} {r['n_rows']:>6d} {r['n_features']:>6d} "
                f"{r['train_accuracy']:>10.3f} {r['test_accuracy']:>9.3f} {r['fpr_gap']:>8.3f}"
            )
        if "raw" in variants and "cleaned" in variants:
            acc_delta = variants["cleaned"]["test_accuracy"] - variants["raw"]["test_accuracy"]
            fpr_delta = variants["cleaned"]["fpr_gap"] - variants["raw"]["fpr_gap"]
            lines.append(
                f"    -> cleaning changed test accuracy by {acc_delta:+.3f} "
                f"and the FPR gap by {fpr_delta:+.3f}"
            )
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


def fairness_report(y_test, y_pred, extras_test: pd.DataFrame, sensitive_attr: str = "race"):
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each race group, prints (and returns as text) the false
    positive rate (share of people who did NOT reoffend but were
    predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text is anything other than "low", case-insensitively, counts as a "high risk" prediction), for comparison

    Returns (text, our_model_fpr_gap), where the gap is the spread
    (max - min) of our model's own false-positive rate across race
    groups on this test set -- a single number that compare_before_after
    can track across the before/after and model comparisons without
    re-parsing the printed text.
    """
    df = extras_test.copy()
    df["y_true"] = y_test.values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"].astype(str).str.lower() != "low").astype(int)

    lines = [
        "False positive rate by race",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    our_model_fprs = []
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
            if col == "y_pred_model":
                our_model_fprs.append(fpr)
        lines.append("")

    text = "\n".join(lines)
    print(text)

    fpr_gap = (max(our_model_fprs) - min(our_model_fprs)) if our_model_fprs else float("nan")
    return text, fpr_gap
