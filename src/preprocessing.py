"""
Preprocessing -- cleaning rules (stateless), the training-only duplicate
drop, the feature/target split, the locked test-set split, and the
leak-safe recipe (imputation + encoding + scaling) that goes inside the
model's sklearn Pipeline.

See the "Preprocessing regime" table in 02_preprocessing.ipynb for the
three kinds of step and where each is allowed to run:
    - stateless rule (this file's clean_dataset / _canonicalize_categories /
      flag_invalid_values): no fitting, safe on training data AND new data
      to predict.
    - training-data decision (drop_duplicate_rows): removes rows, so it
      only runs on the labelled training file, before the dev/test split.
    - fitted step (build_preprocessor's imputers/encoder/scaler): learns
      from data, so it lives inside the Pipeline and is only ever fit on
      the training rows of each split/fold.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (makes IterativeImputer importable)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies a dict of {column: {"min": ..., "max": ...}} domain rules (either bound is
    optional) and converts violations to NaN **in place** on `df`. An "impossible but
    not missing" value (an age of -3, a COMPAS decile score of 15) counts as missing
    once this runs -- `.isna()` alone would never have caught it.

    Returns a small report: how many violations were found per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the week 3 diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, and redundant-column removal. Target-agnostic -- safe to call on
    label-free inference data, since none of this depends on a target column.

    Row-preserving: every input row comes out, in the same order. Removing duplicate
    rows is a *training-only* decision and lives in `drop_duplicate_rows()` -- at
    prediction time every row needs a prediction.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}) or {}, placeholder_tokens)

    # week 9: placeholder tokens ("?", "-", ...) become NaN in EVERY text column, not only in the
    # ones listed in canonical_categories -- on Adult, "?" in workclass/occupation/native_country
    # would otherwise survive as a category called "?" (on COMPAS every such column is listed,
    # so its result is unchanged)
    for col in out.select_dtypes(include=["object", "string"]).columns:
        is_placeholder = out[col].astype("string").str.strip().isin(placeholder_tokens)
        out.loc[is_placeholder.fillna(False).astype(bool), col] = np.nan

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY. Drops exact duplicate rows and repeated ids (keeping the
    first), so the same person can't be counted twice -- or land in both the
    development and the locked test set. Must run *before* `split_dev_test()`.

    Never call this on data you're predicting for: every row there needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR-diagnosed column, before that
    column gets imputed -- so a model can still see the pattern even though the fill
    value itself (median/mode) can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is `None` and `extras` has no target column when called
    on label-free inference data -- nothing downstream requires the target to be present.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])
    # week 9: the external score compared against in the fairness report is a config setting
    # (COMPAS: score_text), not a column name hardcoded here -- Adult has none
    baseline = data_config.get("baseline_score") or {}

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, baseline.get("column")] if c and c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Sets the final test set aside. Stratified split of X, y and the extras frame
    (race/score_text, kept for the fairness report) together, so all three stay
    row-aligned. Returns a *development* set and a *locked test set*:
      - development set: everything we're allowed to learn from and compare models on.
        Cross-validation (src/evaluate.py) splits it again into train/validation folds.
      - locked test set: never used to fit, tune, compare or choose anything. Its size
        and seed live in config.yaml's `test_set` section and are never changed after
        this week.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed),
}


# Week 10: numeric imputers. "iterative" is MICE with one completed dataset; "random_forest" is
# the missForest idea (each column predicted from the others by a random forest). All of them
# learn only from the feature columns, never from the target, so they also run on data to predict.
_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(strategy="median"),
    "mean": lambda cfg, seed: SimpleImputer(strategy="mean"),
    "knn": lambda cfg, seed: KNNImputer(n_neighbors=cfg.get("n_neighbors", 5)),
    "iterative": lambda cfg, seed: IterativeImputer(max_iter=cfg.get("max_iter", 10), random_state=seed),
    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=cfg.get("n_estimators", 50), min_samples_leaf=5,
                                        random_state=seed, n_jobs=-1),
        max_iter=cfg.get("max_iter", 10), random_state=seed),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: scaling and
    imputation for numeric columns (scaled first, so distance-based imputers compare columns
    on one scale; scalers ignore NaN when fitting), imputation and encoding for categorical
    ones, and the missingness flags passed through. Every encoder tolerates unseen categories.
    Nothing is fit here: this object goes *inside* the model's Pipeline (see main.py), so it is
    only ever fit on the training part of each CV fold.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))
    numeric_strategy = imputation.get("numeric_strategy", "median")
    if numeric_strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}. Options: {list(_NUMERIC_IMPUTERS)}")

    numeric_pipeline = Pipeline([
        ("scale", scaler),
        ("impute", _NUMERIC_IMPUTERS[numeric_strategy](imputation, preprocessing_config.get("random_state"))),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])
