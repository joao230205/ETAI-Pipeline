"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Week 4: cleans the data (row-preserving), drops duplicate rows (training
data only, before any split), sets aside a locked test set (never touched
below), and evaluates the model currently configured in config.yaml with
stratified k-fold cross-validation on the development set -- reporting
per-fold accuracy, an out-of-fold classification report, and the fairness
audit on those same out-of-fold predictions.

To compare models, change config.yaml's `model.type` (dummy,
logistic_regression, decision_tree, random_forest) and run again -- each
run's full report is saved to results/.
"""
import yaml

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test, build_preprocessor,
)
from src.model import build_model
from src.evaluate import cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.results import save_run
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_pipeline(config: dict) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])

    df_clean = clean_dataset(df_raw, config["diagnostics"])   # row-preserving
    print(
        f"clean_dataset:       {df_raw.shape} -> {df_clean.shape}   "
        f"(same rows, same order: {df_clean.index.equals(df_raw.index)})"
    )

    n_before_dedup = len(df_clean)
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))   # training data only
    print(f"drop_duplicate_rows: -> {df_clean.shape}   ({n_before_dedup - len(df_clean)} duplicate rows removed)")

    X, y, extras = split_features_target(
        df_clean, config["data"], config["preprocessing"]["mnar_indicator_sources"]
    )
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"]
    )
    print(f"Development set: {len(X_dev)} rows  |  Locked test set: {len(X_test)} rows (not touched by this run)")
    print()

    cv_cfg = config["cv"]
    shuffle = cv_cfg.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=shuffle,
        random_state=cv_cfg.get("random_state") if shuffle else None,
    )
    scoring = cv_cfg.get("scoring", "accuracy")

    pipeline = make_pipeline(config)
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_cfg.get("n_jobs", 1)
    )

    report_sections = [f"### model={config['model']['type']}", ""]
    report_sections.append(cv_report(fold_scores, scoring))
    print()
    report_sections.append(oof_classification_report(y_dev, y_oof))
    report_sections.append(
        fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])
    )

    # Evaluation ends in a model: CV fits and discards 5 models to estimate how good the
    # *recipe* is. The model you'd actually use is the same pipeline refit on every
    # development row (the locked test set still isn't touched here).
    final_model = make_pipeline(config).fit(X_dev, y_dev)
    print(f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.")

    full_report = "\n\n".join(report_sections)
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, full_report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
