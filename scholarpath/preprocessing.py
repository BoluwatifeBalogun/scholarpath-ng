"""Data preprocessing (report Section 3.4.4).

Steps, in the order the report lists them:
  1. inspect            -> inspect()
  2. missing values     -> flagged in clean(), imputed inside the pipeline
  3. duplicate records  -> clean()
  4. inconsistent data  -> clean()
  5. categorical encoding -> build_preprocessor()
  6. feature selection  -> select_features()
  7. separate X and y   -> split_xy()
  8. train/test split   -> modeling.train()

Imputation and encoding live inside the scikit-learn pipeline so they
are learned from the training portion only and are applied identically
at prediction time. No feature scaling is applied: a Decision Tree
splits on thresholds and does not need it.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .grading import GRADING, INSTITUTION_TYPES, tier_from_cgpa
from .schema import (EXCLUDED_COLUMNS, FEATURE_BY_NAME, FEATURE_NAMES,
                     NOMINAL_FEATURES, NUMERIC_FEATURES,
                     ORDINAL_FEATURES, REQUIRED_COLUMNS, RESOURCE_LEVELS,
                     TARGET, TARGET_SOURCE)

INSTITUTION_ALIASES = {
    "university": "University", "uni": "University",
    "polytechnic": "Polytechnic", "poly": "Polytechnic",
    "college of education": "College of Education", "coe": "College of Education",
}


class DatasetError(ValueError):
    """Raised when an uploaded dataset cannot be used."""


def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            "The file is missing these required columns: " + ", ".join(missing) + ".")


def inspect(df: pd.DataFrame) -> dict:
    """Step 1. Describe the raw dataset before anything is changed."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "column_info": [
            {"name": c, "dtype": str(df[c].dtype),
             "missing": int(df[c].isna().sum()),
             "unique": int(df[c].nunique(dropna=True))}
            for c in df.columns
        ],
        "missing_total": int(df.isna().sum().sum()),
    }


def _title_match(series: pd.Series, allowed):
    lookup = {a.lower(): a for a in allowed}
    cleaned = series.astype("string").str.strip().str.lower().map(lookup)
    return cleaned


def clean(df: pd.DataFrame):
    """Steps 2 to 4. Return (clean_df, report).

    Nothing is silently invented here: inconsistent spellings are mapped
    to their canonical value, impossible numbers become missing (to be
    imputed later), and rows that cannot be used at all are removed.
    """
    check_columns(df)
    df = df.copy()
    report = {"rows_in": int(len(df))}

    # 4a. Inconsistent category spellings
    raw_itype = df["institution_type"].astype("string").str.strip()
    fixed_itype = raw_itype.str.lower().map(INSTITUTION_ALIASES)
    report["institution_spellings_fixed"] = int(
        (fixed_itype.notna() & (fixed_itype != raw_itype)).sum())
    df["institution_type"] = fixed_itype

    fixed_cats = 0
    for col, allowed in (("learning_resources", RESOURCE_LEVELS),):
        raw = df[col].astype("string")
        fixed = _title_match(raw, allowed)
        fixed_cats += int((fixed.notna() & (fixed != raw)).sum())
        df[col] = fixed
    report["category_spellings_fixed"] = fixed_cats

    # 4b. Wrong formats: force numeric columns to numbers
    for col in NUMERIC_FEATURES + [TARGET_SOURCE]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Duplicate records
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))

    # Rows that cannot be used: no institution type or no outcome
    before = len(df)
    df = df[df["institution_type"].notna() & df[TARGET_SOURCE].notna()]
    report["rows_without_target_removed"] = int(before - len(df))

    # 4c. Out-of-range values become missing
    scale = df["institution_type"].map(lambda t: GRADING[t]["scale"])
    invalid = 0
    for col in NUMERIC_FEATURES:
        f = FEATURE_BY_NAME[col]
        hi = scale if col == "previous_gpa" else f["max"]
        bad = df[col].notna() & ((df[col] < f["min"]) | (df[col] > hi))
        invalid += int(bad.sum())
        df.loc[bad, col] = np.nan
    report["invalid_values_set_missing"] = invalid

    before = len(df)
    df = df[(df[TARGET_SOURCE] >= 0) & (df[TARGET_SOURCE] <= scale.loc[df.index])]
    report["rows_with_invalid_target_removed"] = int(before - len(df))

    # 2. Missing values that remain are imputed inside the pipeline
    report["missing_to_impute"] = {
        c: int(df[c].isna().sum()) for c in FEATURE_NAMES if df[c].isna().any()}
    report["missing_to_impute_total"] = int(sum(report["missing_to_impute"].values()))

    df = df.reset_index(drop=True)
    report["rows_out"] = int(len(df))
    return df, report


def derive_target(df: pd.DataFrame) -> pd.Series:
    """Label every record with its performance tier from the grading structure."""
    return pd.Series(
        [tier_from_cgpa(t, g) for t, g in zip(df["institution_type"], df[TARGET_SOURCE])],
        index=df.index, name=TARGET).astype(int)


def select_features(df: pd.DataFrame):
    """Step 6. Keep the model inputs; report what was left out and why."""
    dropped = [{"column": c, "reason": EXCLUDED_COLUMNS.get(c, "Not a model input.")}
               for c in df.columns if c not in FEATURE_NAMES]
    return df[FEATURE_NAMES].copy(), dropped


def split_xy(df: pd.DataFrame):
    """Step 7. Separate input features from the target."""
    X, dropped = select_features(df)
    y = derive_target(df)
    return X, y, dropped


class GpaToScalePercent(BaseEstimator, TransformerMixin):
    """Express previous GPA as a percentage of the institution's scale.

    A 3.20 means different things on a 4-point and a 5-point scale, so
    the raw number is not comparable across institution types. This
    feature-preparation step puts every record on one 0-100 footing.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        scale = X["institution_type"].map(
            lambda t: GRADING[t]["scale"] if t in GRADING else np.nan)
        X["previous_gpa"] = (X["previous_gpa"] / scale * 100).round(2)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def build_preprocessor() -> Pipeline:
    """Steps 2 and 5: imputation and categorical encoding."""
    numeric = SimpleImputer(strategy="median")
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[RESOURCE_LEVELS])),
    ])
    nominal = OneHotEncoder(categories=[INSTITUTION_TYPES], handle_unknown="ignore",
                            sparse_output=False)
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("ord", ordinal, ORDINAL_FEATURES),
        ("nom", nominal, NOMINAL_FEATURES),
    ], verbose_feature_names_out=False)
    return Pipeline([("gpa", GpaToScalePercent()), ("columns", columns)])


def encoded_feature_names():
    """Column order produced by build_preprocessor()."""
    return (NUMERIC_FEATURES + ORDINAL_FEATURES
            + [f"institution_type_{t}" for t in INSTITUTION_TYPES])


def to_frame(record: dict) -> pd.DataFrame:
    """Turn one validated record into the frame layout used in training."""
    return pd.DataFrame([{name: record[name] for name in FEATURE_NAMES}])
