"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Runs the same raw data through two preprocessing paths -- the naive
`preprocess()` on its own ("raw") and `clean_dataset()` (src/cleaning.py)
followed by `preprocess()` ("cleaned") -- and, on each path, through every
model listed in config.yaml's `models` section. All four (dataset x model)
runs are evaluated the same way and summarized side by side.
"""
import yaml
from sklearn.metrics import accuracy_score

from src.data import load_data
from src.cleaning import clean_dataset
from src.preprocessing import preprocess
from src.model import build_model
from src.evaluate import evaluate, fairness_report, compare_before_after
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_variant(df, config, dataset_label: str, model_name: str, model_config: dict, report_sections: list) -> dict:
    """Runs one (dataset variant, model) combination end to end: split,
    train, evaluate, fairness-audit. Appends the human-readable report for
    this combination to `report_sections` and returns the numeric summary
    `compare_before_after` needs."""
    X_train, X_test, y_train, y_test, extras_test = preprocess(
        df,
        features=config["data"]["features"],
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        extra_audit_columns=config["data"]["extra_audit_columns"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    model = build_model(model_config)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    section = [f"### dataset={dataset_label}  model={model_name}", ""]
    section.append(evaluate(y_train, y_train_pred, y_test, y_test_pred))
    fairness_text, fpr_gap = fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )
    section.append(fairness_text)
    report_sections.append("\n".join(section))

    return {
        "dataset": dataset_label,
        "model": model_name,
        # actual rows used for train+test, after preprocess()'s dropna --
        # not len(df), which would still count rows preprocess() discards
        "n_rows": X_train.shape[0] + X_test.shape[0],
        "n_features": X_train.shape[1],
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "fpr_gap": fpr_gap,
    }


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw)

    print(
        f"clean_dataset: {len(df_raw)} rows -> {len(df_clean)} rows "
        f"({len(df_raw) - len(df_clean)} dropped -- duplicates, missing sex/race/c_charge_degree), "
        f"{df_raw.shape[1]} columns -> {df_clean.shape[1]} columns "
        f"({df_raw.shape[1] - df_clean.shape[1]} redundant columns dropped)"
    )
    print()

    report_sections = []
    summaries = []
    for dataset_label, df in [("raw", df_raw), ("cleaned", df_clean)]:
        for model_name, model_config in config["models"].items():
            summaries.append(
                run_variant(df, config, dataset_label, model_name, model_config, report_sections)
            )

    comparison_text = compare_before_after(summaries)

    full_report = "\n\n".join(report_sections) + "\n\n" + comparison_text

    results_dir = config.get("output", {}).get("results_dir", "results")
    model_names = ", ".join(config["models"].keys())
    run_label = f"Before/after comparison -- datasets: raw, cleaned -- models: {model_names}"
    path = save_run(results_dir, config, full_report, run_label=run_label)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
