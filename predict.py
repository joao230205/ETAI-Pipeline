"""
Predicts a CSV of new people with the model saved by main.py -- no training happens here.

Run with (from the project folder):
    python main.py                                                    # train, evaluate, save the model
    python predict.py data/compas_new_people.csv predictions.csv      # predict new people
    python predict.py data/adult_new.csv predictions_adult.csv --config config_adult.yaml

The new file is cleaned with exactly the same rules as the training data (clean_dataset), but
WITHOUT drop_duplicate_rows(): every input row gets a prediction, so rows out = rows in. Gaps are
filled and categories encoded by the values the saved pipeline learned in training; categories
never seen in training don't crash the encoders.
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Returns the id, the probability of the positive class and the prediction (at 0.5) for every row."""
    clean = clean_dataset(df, config["diagnostics"])
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    probability = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"id": df[config["diagnostics"]["id_column"]],
                         "probability": probability,
                         "prediction": (probability >= 0.5).astype(int)})


def check_columns(df: pd.DataFrame, config: dict) -> None:
    """Stops with a clear message if the input lacks a column the model needs."""
    prep = config["preprocessing"]
    needed = set(prep["numeric_features"]) | set(prep["categorical_features"]) | {config["diagnostics"]["id_column"]}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise SystemExit(f"The input file is missing columns the model needs: {', '.join(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="the config the model was trained with")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    new_data = pd.read_csv(args.input)
    check_columns(new_data, config)
    pipeline = joblib.load(config["output"]["model_path"])
    predictions = predict(new_data, pipeline, config)
    assert len(predictions) == len(new_data), "every input row must get exactly one prediction"
    predictions.to_csv(args.output, index=False)
    print(f"{len(new_data)} rows in, {len(predictions)} rows predicted -> {args.output}")
