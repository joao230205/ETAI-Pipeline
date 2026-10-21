"""
Entry point for the predictive pipeline.

Run with:
    python main.py                      # uses config.yaml
    python main.py --config other.yaml  # same, with another config file

Weeks 4-7: cleans the data (row-preserving), drops duplicate rows (training data only, before
any split), sets aside a locked test set (never touched below), and evaluates the model
configured in config.yaml on the development set:
  - the floor (a dummy model) and the model with its configured hyperparameters, by stratified
    k-fold cross-validation of the whole Pipeline (preprocessing + model);
  - if `tuning.enabled`: Optuna tuning, scored honestly with nested cross-validation;
  - several metrics per fold, out-of-fold probabilities, a classification report and the
    fairness audit on those same out-of-fold predictions;
  - one row appended to the scoreboard (results/runs.csv), the full report saved to results/.

To compare models, change config.yaml's `model.type` and run again.
"""
import argparse
from datetime import datetime

import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import build_scorers, cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_scores, save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    config = load_config(config_path)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    metrics, selection_metric = cv_cfg["metrics"], cv_cfg["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_cfg.get("n_jobs", 1)
    model_type = config["model"]["type"]

    # the floor: what a model that always predicts the majority class scores on the same folds
    dummy_config = {"type": "dummy", "params": {}}
    floor_scores, _ = cross_validate_pipeline(
        build_pipeline(config["preprocessing"], dummy_config), X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    floor = floor_scores["validation"].mean()

    # the model with the hyperparameters written in config.yaml (the "untuned" score)
    pipeline = build_pipeline(config["preprocessing"], config["model"])
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    untuned = fold_scores["validation"].mean()

    report_sections = [f"### model={model_type}", ""]
    report_sections.append(cv_report(fold_scores, metrics, selection_metric,
                                     title="Cross-validation, configured hyperparameters"))

    tuning_cfg = config.get("tuning", {})
    tuned = bool(tuning_cfg.get("enabled")) and model_type != "dummy"
    tuned_params = {}
    if tuned:
        search_space = tuning_cfg.get("search_spaces", {}).get(model_type)
        if search_space is None:
            raise ValueError(f"tuning is enabled but tuning.search_spaces has no entry for '{model_type}'")
        inner_cv = StratifiedKFold(n_splits=tuning_cfg.get("n_splits", cv_cfg["n_splits"]), shuffle=True,
                                   random_state=tuning_cfg["random_state"])
        n_trials, seed = tuning_cfg["n_trials"], tuning_cfg["random_state"]
        print(f"\nTuning {model_type}: {n_trials} trials, nested CV ({cv_cfg['n_splits']} outer x "
              f"{inner_cv.get_n_splits()} inner folds)...")

        # honest estimate: tuning repeated inside every outer fold
        fold_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, seed, n_jobs)
        # the hyperparameters to keep: tuned once on all development rows
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, seed, n_jobs)
        tuned_params = study.best_params
        report_sections.append(cv_report(
            fold_scores.drop(columns=["inner_best"] + list(search_space)), metrics, selection_metric,
            title="Nested cross-validation (honest score of the tuned model)"))
        report_sections.append(tuning_report(study, fold_scores, selection_metric))

    report_sections.append(oof_classification_report(y_dev, oof_proba))
    report_sections.append(
        fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])
    )
    print("\n\n".join(report_sections[2:]))
    print()

    # Evaluation ends in a model: CV fits and discards 5 models to estimate how good the
    # *recipe* is. The model you'd actually use is the same pipeline refit on every
    # development row (the locked test set still isn't touched here).
    final_model = pipeline.fit(X_dev, y_dev)
    print(f"Final model: {model_type} refit on all {len(X_dev)} development rows.")

    # scoreboard row: mean/std/folds are the honest score (nested CV if tuned)
    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuned,
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": config["model"].get("params") or {},
        "tuned_params": tuned_params,
        "metric": selection_metric,
        "mean": fold_scores["validation"].mean(),
        "std": fold_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
    }
    for i, v in enumerate(fold_scores["validation"], start=1):
        row[f"fold_{i}"] = v
    board = save_scores(config["output"]["scoreboard"], row)
    print(f"\nScoreboard ({selection_metric}, higher is better):")
    print(board[["run", "model", "tuned", "mean", "std", "untuned", "floor"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, "\n\n".join(report_sections))
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
