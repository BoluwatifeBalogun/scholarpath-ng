import json

from flask import (Blueprint, Response, abort, flash, g, jsonify, redirect,
                   render_template, request, url_for)

from .. import login_required, modeling, services
from ..db import Prediction
from ..grading import GRADING, TIER_MEANING, bands_table, label_for
from ..pdf import prediction_pdf
from ..schema import FEATURES, FEATURE_BY_NAME, GROUPS, validate_input

bp = Blueprint("predict", __name__)

SAMPLES = {
    "strong": dict(institution_type="University", year_of_study=3, utme_score=268,
                   previous_gpa=4.12, ca_score=24.5, test_score=74, carryovers=0,
                   attendance_rate=92, study_hours=18, assignment_completion=96,
                   learning_resources="High", library_hours=7),
    "at_risk": dict(institution_type="Polytechnic", year_of_study=2, utme_score=188,
                    previous_gpa=2.05, ca_score=13, test_score=41, carryovers=3,
                    attendance_rate=58, study_hours=5, assignment_completion=60,
                    learning_resources="Low", library_hours=1),
}


def _form_config():
    return {"features": FEATURES,
            "scales": {t: g_["scale"] for t, g_ in GRADING.items()},
            "years": {t: g_["years"] for t, g_ in GRADING.items()},
            "samples": SAMPLES}


@bp.route("/predict", methods=["GET", "POST"])
@login_required
def new():
    mv = services.active_model(g.db)
    values, errors = {}, {}
    if request.method == "POST":
        values = {f["name"]: request.form.get(f["name"], "") for f in FEATURES}
        values["reference"] = request.form.get("reference", "")
        clean, errors = validate_input(values)
        if mv is None:
            errors["_form"] = "No trained model is active yet. Ask the administrator to train one."
        if not errors:
            p = services.make_prediction(g.db, g.user, clean, values["reference"])
            return redirect(url_for("predict.result", pid=p.id))
    return render_template("predict.html", values=values, errors=errors, groups=GROUPS,
                           features=FEATURES, model=mv, by_name=FEATURE_BY_NAME,
                           config_json=json.dumps(_form_config())), (422 if errors else 200)


@bp.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """JSON version of the same flow: validate, prepare, classify, return."""
    payload = request.get_json(silent=True) or {}
    clean, errors = validate_input(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 422
    if services.active_model(g.db) is None:
        return jsonify({"ok": False, "errors": {"_form": "No trained model is active."}}), 503
    p = services.make_prediction(g.db, g.user, clean, str(payload.get("reference", "")))
    return jsonify({"ok": True, "id": p.id, "tier": p.tier, "label": p.label,
                    "confidence": p.confidence, "path": p.result["path"],
                    "probabilities": p.result["probabilities"],
                    "url": url_for("predict.result", pid=p.id)})


@bp.route("/api/whatif", methods=["POST"])
@login_required
def api_whatif():
    """Classify a changed record without saving it, for the what-if panel."""
    payload = request.get_json(silent=True) or {}
    clean, errors = validate_input(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 422
    mv = services.active_model(g.db)
    if mv is None:
        return jsonify({"ok": False, "errors": {"_form": "No trained model is active."}}), 503
    out = modeling.predict(services.load_pipeline(mv, g.db), clean)
    return jsonify({"ok": True, "tier": out["tier"],
                    "label": label_for(clean["institution_type"], out["tier"]),
                    "confidence": out["confidence"], "leaf_samples": out["leaf_samples"]})


def _own(pid):
    p = g.db.get(Prediction, pid)
    if p is None:
        abort(404)
    if p.user_id != g.user.id and not g.user.is_admin:
        abort(404)
    return p


def _view_model(p):
    result, itype = p.result, p.institution_type
    probs = [{"tier": t, "label": label_for(itype, t),
              "p": result["probabilities"].get(str(t), 0.0)} for t in (1, 2, 3, 4, 5)]
    inputs = p.inputs
    rows = []
    for f in FEATURES:
        v = inputs[f["name"]]
        unit = f.get("unit", "")
        shown = f"{v:g}" if isinstance(v, (int, float)) else v
        if unit == "%":
            shown += "%"
        elif unit.startswith("/"):
            shown += " of " + unit[1:]
        elif unit:
            shown += " " + unit
        if f["name"] == "previous_gpa":
            shown += f" of {GRADING[itype]['scale']:g}"
        rows.append({"label": f["label"], "value": shown})
    return {"probs": probs, "rows": rows, "bands": bands_table(itype),
            "meaning": TIER_MEANING[p.tier]}


@bp.route("/predictions/<int:pid>")
@login_required
def result(pid):
    p = _own(pid)
    return render_template("result.html", p=p, r=p.result, vm=_view_model(p))


@bp.route("/predictions/<int:pid>/report.pdf")
@login_required
def pdf(pid):
    p = _own(pid)
    data = prediction_pdf(p, _view_model(p))
    name = f"scholarpath-prediction-{p.id}.pdf"
    return Response(data, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@bp.route("/predictions/<int:pid>/delete", methods=["POST"])
@login_required
def delete(pid):
    p = _own(pid)
    g.db.delete(p)
    g.db.commit()
    flash("Prediction deleted.", "info")
    return redirect(url_for("predict.history"))


@bp.route("/history")
@login_required
def history():
    q = g.db.query(Prediction)
    if not g.user.is_admin:
        q = q.filter_by(user_id=g.user.id)
    tier = request.args.get("tier", type=int)
    if tier in (1, 2, 3, 4, 5):
        q = q.filter_by(tier=tier)
    items = q.order_by(Prediction.created_at.desc()).limit(300).all()
    return render_template("history.html", items=items, tier=tier)
