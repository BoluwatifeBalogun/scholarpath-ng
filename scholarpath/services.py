"""Glue between the web layer, the database and the ML modules."""
import io
import json

import pandas as pd
from werkzeug.security import generate_password_hash

from . import config, modeling
from .db import Dataset, ModelVersion, Prediction, User
from .grading import label_for
from .preprocessing import DatasetError, check_columns, clean, inspect
from .recommend import recommend

_model_cache = {}


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        # keep_default_na=False would hide blanks; only treat empty cells as missing
        df = pd.read_csv(io.BytesIO(data), na_values=[""], keep_default_na=False)
    except Exception as exc:  # malformed file
        raise DatasetError("The file could not be read as CSV. Save it as a "
                           "comma-separated .csv file and try again.") from exc
    if df.empty:
        raise DatasetError("The file has no rows.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def add_dataset(db, name, data: bytes, note="", simulated=False, user_id=None) -> Dataset:
    """Inspect and clean a dataset, then store it with its preprocessing report."""
    df = read_csv_bytes(data)
    check_columns(df)
    inspection = inspect(df)
    cleaned, cleaning = clean(df)
    if len(cleaned) < 200:
        raise DatasetError(f"Only {len(cleaned)} usable records remain after cleaning. "
                           "At least 200 are needed.")
    summary = {
        "by_type": cleaned["institution_type"].value_counts().to_dict(),
        "by_zone": cleaned["zone"].value_counts().to_dict() if "zone" in cleaned else {},
        "states": int(cleaned["state"].nunique()) if "state" in cleaned else 0,
    }
    ds = Dataset(name=name.strip()[:160] or "Untitled dataset", note=note.strip(),
                 simulated=simulated, csv=data, rows_raw=len(df), rows_clean=len(cleaned),
                 report_json=json.dumps({"inspection": inspection, "cleaning": cleaning,
                                         "summary": summary}),
                 uploaded_by=user_id)
    db.add(ds)
    db.commit()
    return ds


def dataset_frame(ds: Dataset) -> pd.DataFrame:
    return read_csv_bytes(ds.csv)


def train_model(db, ds: Dataset, params=None, tune=False, user_id=None,
                activate=False) -> ModelVersion:
    result = modeling.train(dataset_frame(ds), params=params, tune=tune)
    pipeline = result.pop("pipeline")
    last = db.query(ModelVersion).order_by(ModelVersion.version.desc()).first()
    mv = ModelVersion(
        version=(last.version + 1) if last else 1, dataset_id=ds.id,
        params_json=json.dumps(result["params"]), report_json=json.dumps(result),
        accuracy=result["metrics"]["accuracy"], macro_f1=result["metrics"]["macro"]["f1"],
        blob=modeling.dumps(pipeline), trained_by=user_id)
    db.add(mv)
    db.commit()
    _model_cache[mv.id] = pipeline
    if activate:
        activate_model(db, mv)
    return mv


def activate_model(db, mv: ModelVersion):
    """Integrate a trained model: it becomes the one used for predictions."""
    db.query(ModelVersion).update({ModelVersion.is_active: False})
    mv.is_active = True
    db.commit()


def active_model(db):
    return db.query(ModelVersion).filter_by(is_active=True).first()


def load_pipeline(mv: ModelVersion, db=None):
    """Load the saved Joblib pipeline for a model version.

    A Joblib file is tied to the scikit-learn version that wrote it. If
    the library has been upgraded since and the file no longer loads, the
    model is rebuilt from its own dataset and settings (training is
    deterministic, so the result is the same tree) and saved again.
    """
    if mv.id not in _model_cache:
        try:
            _model_cache[mv.id] = modeling.loads(mv.blob)
        except Exception:
            if db is None:
                raise
            result = modeling.train(dataset_frame(mv.dataset), params=mv.params)
            mv.blob = modeling.dumps(result["pipeline"])
            db.commit()
            _model_cache[mv.id] = result["pipeline"]
    return _model_cache[mv.id]


def make_prediction(db, user, record: dict, reference="") -> Prediction:
    mv = active_model(db)
    if mv is None:
        raise RuntimeError("No trained model is active yet.")
    out = modeling.predict(load_pipeline(mv, db), record)
    label = label_for(record["institution_type"], out["tier"])
    out["recommendations"] = recommend(
        record, out["tier"], {s["feature"] for s in out["path"]})
    p = Prediction(user_id=user.id, model_id=mv.id, reference=reference.strip()[:120],
                   institution_type=record["institution_type"],
                   inputs_json=json.dumps(record), tier=out["tier"], label=label,
                   confidence=out["confidence"], result_json=json.dumps(out))
    db.add(p)
    db.commit()
    return p


def seed(db):
    """First run: create the administrator, load the dataset, train the model."""
    if not db.query(User).filter_by(role="admin").first():
        db.add(User(name=config.ADMIN_NAME, email=config.ADMIN_EMAIL.lower(),
                    password_hash=generate_password_hash(config.ADMIN_PASSWORD),
                    role="admin", occupation="System administrator"))
        db.commit()
    if not db.query(Dataset).first() and config.BUNDLED_DATASET.exists():
        add_dataset(db, config.BUNDLED_DATASET_NAME, config.BUNDLED_DATASET.read_bytes(),
                    note=config.BUNDLED_DATASET_NOTE, simulated=True)
    if not db.query(ModelVersion).first():
        ds = db.query(Dataset).order_by(Dataset.id).first()
        if ds:
            train_model(db, ds, activate=True)
