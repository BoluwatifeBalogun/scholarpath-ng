"""Administrator use cases: upload/manage dataset, prepare dataset,
train the Decision Tree, evaluate it, integrate or update the model."""
import io

from flask import (Blueprint, Response, abort, flash, g, redirect,
                   render_template, request, url_for)

from .. import admin_required, services
from ..db import Dataset, ModelVersion, Prediction, User
from ..grading import GRADING, TIERS
from ..modeling import DEFAULT_PARAMS, TUNING_GRID
from ..preprocessing import DatasetError
from ..schema import EXCLUDED_COLUMNS, FEATURES, REQUIRED_COLUMNS, TARGET_SOURCE

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("/")
@admin_required
def overview():
    mv = services.active_model(g.db)
    recent = g.db.query(Prediction).order_by(Prediction.created_at.desc()).limit(6).all()
    tier_counts = {t: g.db.query(Prediction).filter_by(tier=t).count() for t in TIERS}
    return render_template(
        "admin/overview.html", model=mv, report=mv.report if mv else None,
        users=g.db.query(User).order_by(User.created_at.desc()).limit(8).all(),
        counts={"users": g.db.query(User).count(),
                "datasets": g.db.query(Dataset).count(),
                "models": g.db.query(ModelVersion).count(),
                "predictions": g.db.query(Prediction).count()},
        recent=recent, tier_counts=tier_counts)


# ---- datasets --------------------------------------------------------------

@bp.route("/datasets", methods=["GET", "POST"])
@admin_required
def datasets():
    error = None
    if request.method == "POST":
        file = request.files.get("file")
        name = request.form.get("name", "").strip()
        if not file or not file.filename:
            error = "Choose a CSV file to upload."
        elif not file.filename.lower().endswith(".csv"):
            error = "Only .csv files are accepted. Export the sheet as CSV and try again."
        else:
            try:
                ds = services.add_dataset(
                    g.db, name or file.filename.rsplit(".", 1)[0], file.read(),
                    note=request.form.get("note", ""), user_id=g.user.id)
                flash(f"Uploaded and checked. {ds.rows_clean:,} of {ds.rows_raw:,} rows "
                      "can be used.", "success")
                return redirect(url_for("admin.dataset", did=ds.id))
            except DatasetError as exc:
                error = str(exc)
    items = g.db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return render_template("admin/datasets.html", items=items, error=error,
                           required=REQUIRED_COLUMNS, features=FEATURES,
                           target=TARGET_SOURCE), (422 if error else 200)


@bp.route("/datasets/template.csv")
@admin_required
def dataset_template():
    header = ["student_id", "gender", "state", "zone"] + REQUIRED_COLUMNS
    rows = [
        ["EX001", "Female", "Lagos", "South West", "University", 3, 268, 4.12, 24.5, 74, 0,
         92, 18, 96, 7, "High", 4.21],
        ["EX002", "Male", "Kano", "North West", "Polytechnic", 2, 188, 2.05, 13, 41, 3,
         58, 5, 60, 1, "Low", 1.88],
        ["EX003", "Female", "Enugu", "South East", "College of Education", 1, 176, 3.1, 19, 57, 0,
         80, 11, 85, 3, "Medium", 3.02],
    ]
    text = "\n".join(",".join(str(c) for c in r) for r in [header] + rows) + "\n"
    return Response(text, mimetype="text/csv", headers={
        "Content-Disposition": 'attachment; filename="scholarpath-dataset-template.csv"'})


def _dataset(did):
    ds = g.db.get(Dataset, did)
    if ds is None:
        abort(404)
    return ds


@bp.route("/datasets/<int:did>")
@admin_required
def dataset(did):
    ds = _dataset(did)
    df = services.dataset_frame(ds)
    preview = df.head(10).fillna("").astype(str)
    return render_template("admin/dataset.html", ds=ds, report=ds.report,
                           columns=list(preview.columns), rows=preview.values.tolist(),
                           excluded=EXCLUDED_COLUMNS, features=FEATURES,
                           models=g.db.query(ModelVersion).filter_by(dataset_id=ds.id).all())


@bp.route("/datasets/<int:did>/download")
@admin_required
def dataset_download(did):
    ds = _dataset(did)
    return Response(ds.csv, mimetype="text/csv", headers={
        "Content-Disposition": f'attachment; filename="dataset-{ds.id}.csv"'})


@bp.route("/datasets/<int:did>/delete", methods=["POST"])
@admin_required
def dataset_delete(did):
    ds = _dataset(did)
    if g.db.query(ModelVersion).filter_by(dataset_id=ds.id).count():
        flash("This dataset has trained models. Delete those models first.", "error")
        return redirect(url_for("admin.dataset", did=ds.id))
    g.db.delete(ds)
    g.db.commit()
    flash("Dataset deleted.", "info")
    return redirect(url_for("admin.datasets"))


# ---- models ----------------------------------------------------------------

@bp.route("/models", methods=["GET", "POST"])
@admin_required
def models():
    error = None
    datasets_ = g.db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    form = dict(DEFAULT_PARAMS, dataset_id=datasets_[0].id if datasets_ else None,
                tune=False, test_pct=int(DEFAULT_PARAMS["test_size"] * 100))
    if request.method == "POST":
        try:
            ds = g.db.get(Dataset, request.form.get("dataset_id", type=int))
            if ds is None:
                raise ValueError("Choose a dataset to train on.")
            tune = request.form.get("tune") == "on"
            params = {
                "criterion": request.form.get("criterion", "gini"),
                "max_depth": request.form.get("max_depth", type=int),
                "min_samples_leaf": request.form.get("min_samples_leaf", type=int),
                "test_size": (request.form.get("test_pct", type=float) or 20) / 100,
            }
            if tune:
                # the search chooses these three, so whatever was typed is ignored
                params.update(criterion=DEFAULT_PARAMS["criterion"],
                              max_depth=DEFAULT_PARAMS["max_depth"],
                              min_samples_leaf=DEFAULT_PARAMS["min_samples_leaf"])
            form.update(params, dataset_id=ds.id, tune=tune,
                        test_pct=int(params["test_size"] * 100))
            if params["max_depth"] is None or params["min_samples_leaf"] is None:
                raise ValueError("Maximum depth and minimum records per leaf must be whole numbers.")
            mv = services.train_model(g.db, ds, params=params, tune=tune, user_id=g.user.id)
            flash(f"Version {mv.version} trained, {mv.accuracy * 100:.1f}% accurate on test records. "
                  "Read the evaluation, then put it in use if you are satisfied.", "success")
            return redirect(url_for("admin.model", mid=mv.id))
        except (ValueError, DatasetError) as exc:
            error = str(exc)
    items = g.db.query(ModelVersion).order_by(ModelVersion.version.desc()).all()
    return render_template("admin/models.html", items=items, datasets=datasets_,
                           error=error, form=form, grid=TUNING_GRID), (422 if error else 200)


def _model(mid):
    mv = g.db.get(ModelVersion, mid)
    if mv is None:
        abort(404)
    return mv


@bp.route("/models/<int:mid>")
@admin_required
def model(mid):
    mv = _model(mid)
    report = mv.report
    cm = report["metrics"]["confusion_matrix"]
    row_totals = [sum(r) or 1 for r in cm]
    return render_template("admin/model.html", mv=mv, report=report, cm=cm,
                           row_totals=row_totals, tiers=TIERS, grading=GRADING,
                           used=g.db.query(Prediction).filter_by(model_id=mv.id).count())


@bp.route("/models/<int:mid>/activate", methods=["POST"])
@admin_required
def model_activate(mid):
    mv = _model(mid)
    services.activate_model(g.db, mv)
    flash(f"Version {mv.version} is now in use for every new prediction.", "success")
    return redirect(url_for("admin.model", mid=mv.id))


@bp.route("/models/<int:mid>/rules.txt")
@admin_required
def model_rules(mid):
    mv = _model(mid)
    return Response(mv.report["tree"]["rules"], mimetype="text/plain", headers={
        "Content-Disposition": f'attachment; filename="decision-rules-v{mv.version}.txt"'})


@bp.route("/models/<int:mid>/model.joblib")
@admin_required
def model_download(mid):
    mv = _model(mid)
    return Response(io.BytesIO(mv.blob).getvalue(), mimetype="application/octet-stream",
                    headers={"Content-Disposition":
                             f'attachment; filename="decision-tree-v{mv.version}.joblib"'})


@bp.route("/models/<int:mid>/delete", methods=["POST"])
@admin_required
def model_delete(mid):
    mv = _model(mid)
    if mv.is_active:
        flash("The version in use cannot be deleted. Put another version in use first.", "error")
    elif g.db.query(Prediction).filter_by(model_id=mv.id).count():
        flash("This version has answered predictions, so it is kept for the record.", "error")
    else:
        g.db.delete(mv)
        g.db.commit()
        flash(f"Version {mv.version} deleted.", "info")
        return redirect(url_for("admin.models"))
    return redirect(url_for("admin.model", mid=mv.id))
