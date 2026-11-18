"""Unit tests: validation, preprocessing, grading, model (report 3.8, item 1)."""
import numpy as np
import pandas as pd
import pytest

from conftest import VALID
from scholarpath import modeling
from scholarpath.grading import GRADING, bands_table, label_for, tier_from_cgpa
from scholarpath.preprocessing import (DatasetError, build_preprocessor, check_columns,
                                 clean, encoded_feature_names, split_xy, to_frame)
from scholarpath.recommend import recommend
from scholarpath.schema import FEATURE_NAMES, validate_input


# ---- grading ---------------------------------------------------------------

@pytest.mark.parametrize("itype,cgpa,tier,label", [
    ("University", 4.50, 1, "First Class"), ("University", 4.49, 2, "Second Class Upper"),
    ("University", 2.40, 3, "Second Class Lower"), ("University", 1.49, 5, "Fail"),
    ("Polytechnic", 3.50, 1, "Distinction"), ("Polytechnic", 3.49, 2, "Upper Credit"),
    ("Polytechnic", 2.50, 3, "Lower Credit"), ("Polytechnic", 2.00, 4, "Pass"),
    ("Polytechnic", 1.99, 5, "Fail"), ("College of Education", 3.50, 2, "Credit"),
    ("College of Education", 2.40, 3, "Merit"),
])
def test_grading_boundaries(itype, cgpa, tier, label):
    assert tier_from_cgpa(itype, cgpa) == tier
    assert label_for(itype, tier) == label


def test_bands_cover_whole_scale():
    for itype, g in GRADING.items():
        bands = bands_table(itype)
        assert bands[0]["max"] == g["scale"] and bands[-1]["min"] == 0
        assert [b["tier"] for b in bands] == [1, 2, 3, 4, 5]


# ---- input validation ------------------------------------------------------

def test_valid_record_passes():
    clean_, errors = validate_input({k: str(v) for k, v in VALID.items()})
    assert errors == {}
    assert list(clean_) == FEATURE_NAMES
    assert clean_["year_of_study"] == 2 and clean_["previous_gpa"] == 2.9


def test_empty_form_reports_every_field():
    _, errors = validate_input({})
    assert set(errors) == set(FEATURE_NAMES)


@pytest.mark.parametrize("field,value,fragment", [
    ("attendance_rate", "140", "between 0 and 100"),
    ("attendance_rate", "-3", "between 0 and 100"),
    ("study_hours", "abc", "must be a number"),
    ("carryovers", "1.5", "whole number"),
    ("utme_score", "450", "between 100 and 400"),
    ("previous_gpa", "4.6", "between 0 and 4 for a Polytechnic"),
    ("year_of_study", "5", "year 1 to 4"),
    ("learning_resources", "Plenty", "Choose one of"),
    ("institution_type", "Secondary school", "Choose one of"),
    ("ca_score", "nan", "must be a number"),
    ("test_score", "inf", "must be a number"),
])
def test_invalid_values_are_rejected(field, value, fragment):
    record = {k: str(v) for k, v in VALID.items()}
    record[field] = value
    _, errors = validate_input(record)
    assert fragment in errors[field]


def test_gpa_limit_follows_institution_scale():
    record = dict(VALID, previous_gpa="4.6", institution_type="University")
    assert validate_input(record)[1] == {}
    record["institution_type"] = "College of Education"
    record["year_of_study"] = 4
    assert "year 1 to 3" in validate_input(record)[1]["year_of_study"]


# ---- preprocessing ---------------------------------------------------------

def test_cleaning_handles_every_problem(raw_df):
    df, report = clean(raw_df)
    assert report["duplicates_removed"] > 0
    assert report["institution_spellings_fixed"] > 0
    assert report["invalid_values_set_missing"] > 0
    assert report["rows_without_target_removed"] > 0
    assert not df.duplicated().any()
    assert set(df["institution_type"]) == set(GRADING)
    assert df["attendance_rate"].dropna().between(0, 100).all()
    assert df["study_hours"].dropna().ge(0).all()
    assert df["library_hours"].dropna().between(0, 40).all()


def test_missing_columns_are_named():
    with pytest.raises(DatasetError, match="attendance_rate"):
        check_columns(pd.DataFrame({"institution_type": ["University"]}))


def test_target_and_leaky_columns_are_not_features(raw_df):
    df, _ = clean(raw_df)
    X, y, dropped = split_xy(df)
    assert list(X.columns) == FEATURE_NAMES
    assert {"final_cgpa", "gender", "state", "zone", "student_id"} <= {d["column"] for d in dropped}
    assert set(y.unique()) == {1, 2, 3, 4, 5}


def test_preprocessor_encodes_and_imputes(raw_df):
    df, _ = clean(raw_df)
    X, _, _ = split_xy(df)
    out = build_preprocessor().fit_transform(X)
    assert out.shape == (len(X), len(encoded_feature_names()))
    assert not np.isnan(out).any()
    one = build_preprocessor().fit(X).transform(to_frame(VALID))
    names = encoded_feature_names()
    assert one[0, names.index("previous_gpa")] == pytest.approx(72.5)    # 2.9 of 4
    assert one[0, names.index("learning_resources")] == 1                 # Medium
    assert one[0, names.index("institution_type_Polytechnic")] == 1


# ---- model -----------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(raw_df):
    return modeling.train(raw_df)


def test_training_is_reproducible(raw_df, trained):
    again = modeling.train(raw_df)
    assert again["metrics"]["accuracy"] == trained["metrics"]["accuracy"]


def test_metrics_are_consistent(trained):
    m = trained["metrics"]
    cm = np.array(m["confusion_matrix"])
    assert cm.sum() == m["tested"] == trained["split"]["test"]
    assert np.trace(cm) / cm.sum() == pytest.approx(m["accuracy"])
    for i, c in enumerate(m["per_class"]):
        assert cm[i].sum() == c["support"]
        assert c["recall"] == pytest.approx(cm[i, i] / cm[i].sum())
        assert c["precision"] == pytest.approx(cm[i, i] / cm[:, i].sum())
    assert m["accuracy"] > 0.6           # far above the 5-class chance level


def test_tree_respects_overfitting_limits(trained):
    assert trained["tree"]["depth"] <= trained["params"]["max_depth"]
    assert trained["train_accuracy"] - trained["metrics"]["accuracy"] < 0.12


def test_saved_model_predicts_the_same(trained):
    restored = modeling.loads(modeling.dumps(trained["pipeline"]))
    a = modeling.predict(trained["pipeline"], VALID)
    b = modeling.predict(restored, VALID)
    assert a == b
    assert a["tier"] in (1, 2, 3, 4, 5)
    assert sum(a["probabilities"].values()) == pytest.approx(1.0)
    assert a["path"] and all(s["text"] for s in a["path"])


def test_better_record_never_predicts_worse(trained):
    strong = dict(VALID, previous_gpa=3.8, attendance_rate=96, ca_score=27, test_score=85,
                  carryovers=0, study_hours=22, assignment_completion=98)
    weak = dict(VALID, previous_gpa=1.4, attendance_rate=40, ca_score=8, test_score=28,
                carryovers=5, study_hours=2, assignment_completion=35)
    assert (modeling.predict(trained["pipeline"], strong)["tier"]
            < modeling.predict(trained["pipeline"], weak)["tier"])


def test_bad_training_settings_are_refused(raw_df):
    with pytest.raises(ValueError, match="depth"):
        modeling.train(raw_df, {"max_depth": 99})
    with pytest.raises(ValueError, match="200 usable"):
        modeling.train(raw_df.head(60))


def test_recommendations_follow_the_record():
    recs = recommend(dict(VALID, attendance_rate=50, carryovers=3), 4, {"attendance_rate"})
    titles = [r["title"] for r in recs]
    assert titles[0] == "Raise lecture attendance" and recs[0]["on_path"]
    assert "Clear outstanding carryovers" in titles
    assert "Ask for academic support now" in titles
