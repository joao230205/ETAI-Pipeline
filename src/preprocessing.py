"""
Preprocessing -- turns a dataframe (raw or already run through
src/cleaning.py) into train/test splits ready for a model:
    - keeps only the configured feature columns (see config.yaml's
      "features" list and its comments for why each excluded column is excluded)
    - drops rows still missing a value in a feature or the target (on the
      "raw" / naive path this is the only missing-value handling at all;
      on the "cleaned" path this should be a no-op, since clean_dataset
      already resolved every missing value in the feature columns)
    - one-hot encodes the categorical features (sex, c_charge_degree)
    - splits into train/test, stratified on the target

`sensitive_attr` (race) and any `extra_audit_columns` (score_text) are
kept aside for the fairness audit -- split alongside the data so they line
up with the test set, but never used as a model input.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(
    df: pd.DataFrame,
    features: list,
    target: str,
    sensitive_attr: str,
    extra_audit_columns: list,
    test_size: float,
    random_state: int,
):
    audit_columns = [sensitive_attr] + list(extra_audit_columns)
    required_columns = features + [target] + audit_columns
    df = df.dropna(subset=[c for c in required_columns if c in df.columns])

    y = df[target]
    extras = df[audit_columns].copy()  # kept aside for the fairness audit, never a model input

    X = pd.get_dummies(df[features], drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test
