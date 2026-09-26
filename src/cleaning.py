"""
Cleaning -- turning the raw, messy COMPAS CSV into something trustworthy.

Every step here traces back to a specific finding from the EDA notebook
(01_eda_introduction.ipynb / JupyterNootebook_Cleaning.ipynb). The
comments reference *why*, not just *what*, so this stays readable without
the notebook open next to it.

`clean_dataset` runs on the whole dataset (train + test together) because
every step here is either a formatting fix or reads only from the row
itself (or from `age_cat`/`c_charge_degree`, which are known before the
split) -- nothing here is fit on the target or leaks test-set information
into training.
"""
import numpy as np
import pandas as pd

# --- constants, each tied to a specific EDA finding -------------------

# "-" and "?" are used throughout the raw CSV as missing-value markers,
# hiding inside otherwise valid-looking text columns (found in race, sex,
# priors_count, prior_offenses).
PLACEHOLDER_TOKENS = {"-", "?"}

# Same category spelled differently in a way plain .lower().strip()
# doesn't catch (a real alias/abbreviation, not just case/whitespace).
CATEGORY_ALIASES = {
    "c_charge_degree": {"felony": "f", "misdemeanor": "m"},
    "race": {"african american": "african-american"},  # space vs hyphen
}

# COMPAS's own decile score is defined on a 1-10 scale.
DECILE_SCORE_RANGE = (1, 10)

# Plausible adult age range for this dataset.
AGE_RANGE = (18, 100)

# priors_count's real values form a continuous, decreasing distribution
# from 0 to 38. Above that there's a hard gap straight to 250 and 500
# (each appearing exactly 3 times) -- a classic sentinel-value pattern,
# not genuine counts. Any cutoff between 38 and 250 isolates them cleanly;
# 60 was chosen with headroom to spare, not because of an IQR rule (the
# classic IQR rule here would cut off plausible values in the real tail).
PRIORS_COUNT_MAX = 60

# score_text's official definition: Low = 1-4, Medium = 5-7, High = 8-10.
SCORE_TEXT_RANGES = {"low": (1, 4), "medium": (5, 7), "high": (8, 10)}

# Columns with low missingness (<3.3%) and no defensible way to impute --
# sex/race are demographic attributes with no "N/A" case, and inventing a
# value for the sensitive attribute (race) would distort the fairness
# audit downstream. Dropped rather than imputed.
DROP_MISSING_SUBSET = ["sex", "race", "c_charge_degree"]

# priors_count is imputed by group median rather than a single global
# median -- but grouped by age_cat / c_charge_degree, deliberately NOT by
# race: grouping the imputation by the sensitive attribute would manufacture
# a statistical dependency between race and priors_count in the imputed
# values, which could distort the fairness audit before it even starts.
PRIORS_COUNT_IMPUTE_GROUPBY = ["age_cat", "c_charge_degree"]

# Exactly redundant with another column, confirmed by row-by-row comparison:
# prior_offenses == priors_count; age_in_months == age * 12;
# juvenile_total == juv_fel_count + juv_misd_count + juv_other_count.
REDUNDANT_COLUMNS = ["prior_offenses", "age_in_months", "juvenile_total"]


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase every text column, then apply the
    explicit alias maps for the cases .lower().strip() can't catch on its
    own (real word variants, not just case/whitespace)."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    for col, alias_map in CATEGORY_ALIASES.items():
        if col in out.columns:
            out[col] = out[col].replace(alias_map)
    return out


def _placeholders_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Turns "-"/"?" into real NaN wherever they appear, so they stop being
    silently treated as valid categories or valid numbers."""
    out = df.copy()
    for col in out.columns:
        text = out[col].astype("string").str.strip().str.lower()
        out.loc[text.isin(PLACEHOLDER_TOKENS), col] = np.nan
    return out


def _apply_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-range / impossible values -> NaN. Doesn't guess a replacement,
    just flags them as missing so the imputation step (or a defensible drop)
    handles them consistently with every other missing value."""
    out = df.copy()

    decile = pd.to_numeric(out["decile_score"], errors="coerce")
    out.loc[~decile.between(*DECILE_SCORE_RANGE), "decile_score"] = np.nan

    age = pd.to_numeric(out["age"], errors="coerce")
    out.loc[age.notna() & ~age.between(*AGE_RANGE), "age"] = np.nan

    juv_fel = pd.to_numeric(out["juv_fel_count"], errors="coerce")
    out.loc[juv_fel.notna() & (juv_fel < 0), "juv_fel_count"] = np.nan

    priors = pd.to_numeric(out["priors_count"], errors="coerce")
    out.loc[priors.notna() & (priors > PRIORS_COUNT_MAX), "priors_count"] = np.nan

    return out


def _resolve_score_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """decile_score and score_text should agree (Low=1-4, Medium=5-7,
    High=8-10). Where both are present, in-range, and still contradict each
    other, there's no way to tell which of the two is wrong -- so both are
    cleared rather than arbitrarily trusting one. Affects ~9 rows (~0.1%)."""
    out = df.copy()
    decile = pd.to_numeric(out["decile_score"], errors="coerce")

    mismatch = pd.Series(False, index=out.index)
    for label, (low, high) in SCORE_TEXT_RANGES.items():
        this_label = out["score_text"] == label
        mismatch |= this_label & decile.notna() & ~decile.between(low, high)

    out.loc[mismatch, ["decile_score", "score_text"]] = np.nan
    return out


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Exact row duplicates and repeated ids point at the same person
    entered twice -- keep the first occurrence of each."""
    out = df.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")
    return out


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Column-by-column imputation, each choice justified by that column's
    own distribution and role (see the module-level constants above for the
    reasoning behind each one)."""
    out = df.copy()

    # heavily skewed toward 0 (most people have no juvenile record) -- 0 is
    # both the mode and a safe, non-sensitive default
    out["juv_fel_count"] = pd.to_numeric(out["juv_fel_count"], errors="coerce").fillna(0)

    # age: simple global median, no sensitive-attribute concerns here
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["age"] = out["age"].fillna(out["age"].median())

    # priors_count: median within (age_cat, c_charge_degree) groups -- all
    # 6 groups have >300 observations, comfortably enough for a stable
    # median (see PRIORS_COUNT_IMPUTE_GROUPBY's docstring for why race is
    # deliberately excluded from the grouping)
    priors = pd.to_numeric(out["priors_count"], errors="coerce")
    group_median = priors.groupby(
        [out[c] for c in PRIORS_COUNT_IMPUTE_GROUPBY]
    ).transform("median")
    out["priors_count"] = priors.fillna(group_median).fillna(priors.median())

    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full cleaning recipe end to end, in the order each step
    depends on the last:
        1. normalize category spelling (case/whitespace + explicit aliases)
        2. placeholder tokens ("-", "?") -> NaN
        3. domain-rule violations -> NaN
        4. decile_score / score_text mutual-consistency check -> NaN
        5. drop exact duplicate rows / repeated ids
        6. drop rows missing sex, race, or c_charge_degree
        7. impute the remaining missing values (juv_fel_count, age, priors_count)
        8. drop the columns confirmed redundant via multicollinearity

    Returns a cleaned copy; `df` is never modified in place.
    """
    out = df.copy()
    out = _normalize_categories(out)
    out = _placeholders_to_nan(out)
    out = _apply_domain_rules(out)
    out = _resolve_score_mismatches(out)
    out = _drop_duplicates(out)
    out = out.dropna(subset=DROP_MISSING_SUBSET)
    out = _impute(out)
    out = out.drop(columns=[c for c in REDUNDANT_COLUMNS if c in out.columns])
    return out
