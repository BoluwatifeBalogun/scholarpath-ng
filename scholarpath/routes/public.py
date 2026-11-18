from flask import Blueprint, g, jsonify, render_template, request

from .. import modeling, services
from ..grading import label_for
from ..db import Dataset, Prediction
from ..schema import EXCLUDED_COLUMNS, FEATURES, validate_input

bp = Blueprint("public", __name__)


@bp.route("/")
def home():
    mv = services.active_model(g.db)
    ds = g.db.get(Dataset, mv.dataset_id) if mv else None
    return render_template(
        "home.html", model=mv, report=mv.report if mv else None,
        dataset=ds, ds_report=ds.report if ds else None,
        features=FEATURES, excluded=EXCLUDED_COLUMNS,
        prediction_count=g.db.query(Prediction).count())


# Values held fixed in the home-page demo: close to the dataset medians.
DEMO_BASE = dict(year_of_study=2, utme_score=212, carryovers=0, test_score=56,
                 study_hours=12, assignment_completion=79, library_hours=4,
                 learning_resources="Medium")


@bp.route("/api/demo")
def demo():
    """Live prediction for the home page: three inputs vary, the rest stay typical."""
    record = dict(DEMO_BASE,
                  institution_type=request.args.get("type", "University"),
                  previous_gpa=request.args.get("gpa", ""),
                  attendance_rate=request.args.get("attendance", ""),
                  ca_score=request.args.get("ca", ""))
    clean, errors = validate_input(record)
    mv = services.active_model(g.db)
    if errors or mv is None:
        return jsonify({"ok": False}), 422
    out = modeling.predict(services.load_pipeline(mv, g.db), clean)
    return jsonify({"ok": True, "tier": out["tier"],
                    "label": label_for(clean["institution_type"], out["tier"]),
                    "confidence": out["confidence"], "steps": len(out["path"])})


@bp.route("/healthz")
def healthz():
    return {"status": "ok", "model_active": services.active_model(g.db) is not None}
