"""Decision Tree training, evaluation, persistence and prediction.

Covers report Sections 3.5.1 (stages 3 to 5 and 7), 3.6 and 3.7.
"""
import io
import time

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from .grading import GRADING, INSTITUTION_TYPES, TIERS
from .preprocessing import (build_preprocessor, clean, encoded_feature_names,
                            inspect, split_xy, to_frame)
from .schema import FEATURE_BY_NAME, RESOURCE_LEVELS

RANDOM_STATE = 42

# Defaults chosen by the 5-fold grid search below on the bundled dataset
# (python scripts/train.py --tune). Depth and leaf-size limits are the
# guard against overfitting discussed in report Section 2.5.3.
DEFAULT_PARAMS = {
    "criterion": "entropy",
    "max_depth": 11,
    "min_samples_leaf": 10,
    "test_size": 0.20,
}

TUNING_GRID = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [5, 7, 9, 11],
    "tree__min_samples_leaf": [10, 20, 30, 50],
}


def normalise_params(params=None):
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})
    if p["criterion"] not in ("gini", "entropy"):
        raise ValueError("Splitting criterion must be gini or entropy.")
    p["max_depth"] = int(p["max_depth"])
    p["min_samples_leaf"] = int(p["min_samples_leaf"])
    p["test_size"] = float(p["test_size"])
    if not 2 <= p["max_depth"] <= 20:
        raise ValueError("Maximum depth must be between 2 and 20.")
    if not 1 <= p["min_samples_leaf"] <= 200:
        raise ValueError("Minimum records per leaf must be between 1 and 200.")
    if not 0.10 <= p["test_size"] <= 0.40:
        raise ValueError("Test share must be between 10% and 40%.")
    return p


def build_pipeline(params) -> Pipeline:
    tree = DecisionTreeClassifier(
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", build_preprocessor()), ("tree", tree)])


def evaluate(y_true, y_pred) -> dict:
    """Accuracy, per-class precision/recall/F1, averages, confusion matrix."""
    labels = TIERS
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    wrong = yt != yp
    adjacent = float((np.abs(yt - yp)[wrong] == 1).mean()) if wrong.any() else 1.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # share of the errors that miss by exactly one tier
        "adjacent_error_share": adjacent,
        "per_class": [
            {"tier": int(t), "precision": float(p), "recall": float(r),
             "f1": float(f), "support": int(s)}
            for t, p, r, f, s in zip(labels, precision, recall, f1, support)],
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]),
                  "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]),
                     "f1": float(weighted[2])},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "tested": int(len(y_true)),
    }


def train(raw_df, params=None, tune=False) -> dict:
    """Run the whole offline process on a raw dataset.

    Returns a dict with the fitted pipeline, the parameters used, the
    preprocessing report, evaluation on the held-out test set and a
    description of the learned tree.
    """
    started = time.time()
    params = normalise_params(params)
    inspection = inspect(raw_df)
    df, cleaning = clean(raw_df)
    X, y, dropped = split_xy(df)

    counts = y.value_counts()
    if len(X) < 200:
        raise ValueError("At least 200 usable records are needed to train a model.")
    if counts.min() < 5:
        raise ValueError("Every performance class needs at least 5 records; "
                         f"the smallest has {int(counts.min())}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline(params)
    tuning = None
    if tune:
        search = GridSearchCV(
            pipeline, TUNING_GRID, scoring="f1_macro", n_jobs=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        best = search.best_params_
        params.update(criterion=best["tree__criterion"],
                      max_depth=int(best["tree__max_depth"]),
                      min_samples_leaf=int(best["tree__min_samples_leaf"]))
        tuning = {"cv_folds": 5, "scoring": "macro F1",
                  "candidates": int(len(search.cv_results_["params"])),
                  "best_cv_score": float(search.best_score_)}
    else:
        pipeline.fit(X_train, y_train)

    test_metrics = evaluate(y_test, pipeline.predict(X_test))
    train_accuracy = float(accuracy_score(y_train, pipeline.predict(X_train)))

    # Accuracy by institution type on the test set
    pred = pipeline.predict(X_test)
    by_type = []
    for t in INSTITUTION_TYPES:
        mask = (X_test["institution_type"] == t).to_numpy()
        if mask.sum():
            by_type.append({"type": t, "tested": int(mask.sum()),
                            "accuracy": float(accuracy_score(y_test[mask], pred[mask]))})

    tree = pipeline.named_steps["tree"]
    names = encoded_feature_names()
    importances = sorted(
        ({"feature": n, "label": display_name(n), "importance": float(v)}
         for n, v in zip(names, tree.feature_importances_)),
        key=lambda d: -d["importance"])

    return {
        "pipeline": pipeline,
        "params": params,
        "inspection": inspection,
        "cleaning": cleaning,
        "dropped": dropped,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)),
                  "test_size": params["test_size"],
                  "class_counts": {int(k): int(v) for k, v in counts.sort_index().items()}},
        "metrics": test_metrics,
        "train_accuracy": train_accuracy,
        "by_type": by_type,
        "tuning": tuning,
        "tree": {"depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves()),
                 "nodes": int(tree.tree_.node_count),
                 "importances": importances,
                 "structure": tree_structure(tree, max_depth=3),
                 "rules": export_text(tree, feature_names=[display_name(n) for n in names],
                                      max_depth=12, decimals=2)},
        "sklearn_version": sklearn.__version__,
        "seconds": round(time.time() - started, 2),
    }


# ---- persistence -----------------------------------------------------------

def dumps(pipeline) -> bytes:
    """Serialise the fitted pipeline (preprocessing + tree) with Joblib."""
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer, compress=3)
    return buffer.getvalue()


def loads(blob: bytes):
    return joblib.load(io.BytesIO(blob))


# ---- explanation -----------------------------------------------------------

def display_name(encoded: str) -> str:
    if encoded.startswith("institution_type_"):
        return "Institution is " + encoded.replace("institution_type_", "")
    if encoded == "previous_gpa":
        return "Previous GPA (% of scale)"
    return FEATURE_BY_NAME[encoded]["label"]


def _fmt(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def describe_split(encoded, threshold, went_left, record=None):
    """Say a tree test in words a lecturer or student can read."""
    if encoded.startswith("institution_type_"):
        t = encoded.replace("institution_type_", "")
        article = "a " + t
        text = f"Institution is not {article}" if went_left else f"Institution is {article}"
        return {"text": text, "value": record["institution_type"] if record else None}

    if encoded == "learning_resources":
        levels = RESOURCE_LEVELS
        cut = int(np.floor(threshold))
        group = levels[:cut + 1] if went_left else levels[cut + 1:]
        label = FEATURE_BY_NAME[encoded]["label"]
        return {"text": f"{label} is {' or '.join(group)}",
                "value": record[encoded] if record else None}

    f = FEATURE_BY_NAME[encoded]
    op = "at most" if went_left else "above"
    if encoded == "previous_gpa":
        if record and record.get("institution_type") in GRADING:
            scale = GRADING[record["institution_type"]]["scale"]
            text = (f"Previous GPA is {op} {_fmt(threshold / 100 * scale)} "
                    f"on the {scale:g}-point scale")
            return {"text": text, "value": _fmt(record["previous_gpa"])}
        return {"text": f"Previous GPA is {op} {_fmt(threshold)}% of the scale",
                "value": None}

    if f["kind"] == "int":
        shown = int(np.floor(threshold))
        text = (f"{f['label']} is {shown} or fewer" if went_left
                else f"{f['label']} is more than {shown}")
        if encoded == "carryovers":
            if shown == 0:
                text = ("Has no outstanding carryovers" if went_left
                        else "Has at least one outstanding carryover")
            else:
                text = (f"Has {shown} or fewer outstanding carryovers" if went_left
                        else f"Has more than {shown} outstanding carryovers")
        if encoded in ("utme_score", "year_of_study"):
            text = (f"{f['label']} is {shown} or lower" if went_left
                    else f"{f['label']} is above {shown}")
    else:
        unit = "%" if f.get("unit") == "%" else ""
        text = f"{f['label']} is {op} {_fmt(threshold)}{unit}"
    value = None
    if record:
        value = _fmt(float(record[encoded]))
        if f.get("unit") == "%":
            value += "%"
    return {"text": text, "value": value}


def tree_structure(tree, max_depth=3):
    """Top levels of the tree as nested dicts, for drawing."""
    t = tree.tree_
    names = encoded_feature_names()
    classes = [int(c) for c in tree.classes_]

    def node(i, depth):
        counts = t.value[i][0]
        total = float(counts.sum())
        dist = {classes[k]: float(counts[k] / total) for k in range(len(classes))}
        majority = classes[int(np.argmax(counts))]
        out = {"id": int(i), "samples": int(t.n_node_samples[i]),
               "tier": majority, "share": float(max(dist.values()))}
        is_leaf = t.children_left[i] == -1
        if is_leaf:
            out["leaf"] = True
            return out
        out["leaf"] = False
        enc = names[t.feature[i]]
        out["test"] = describe_split(enc, float(t.threshold[i]), True)["text"]
        if depth >= max_depth:
            out["truncated"] = True
            return out
        out["yes"] = node(int(t.children_left[i]), depth + 1)
        out["no"] = node(int(t.children_right[i]), depth + 1)
        return out

    return node(0, 0)


def predict(pipeline, record: dict) -> dict:
    """Classify one validated record and explain the route taken."""
    X = to_frame(record)
    prep, tree = pipeline.named_steps["prep"], pipeline.named_steps["tree"]
    encoded = prep.transform(X)
    proba = tree.predict_proba(encoded)[0]
    classes = [int(c) for c in tree.classes_]
    tier = classes[int(np.argmax(proba))]

    names = encoded_feature_names()
    t = tree.tree_
    path_nodes = tree.decision_path(encoded).indices
    steps = []
    for node_id in path_nodes:
        if t.children_left[node_id] == -1:
            leaf_samples = int(t.n_node_samples[node_id])
            continue
        feat = t.feature[node_id]
        went_left = bool(encoded[0, feat] <= t.threshold[node_id])
        step = describe_split(names[feat], float(t.threshold[node_id]), went_left, record)
        step["feature"] = names[feat]
        steps.append(step)

    return {
        "tier": tier,
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "confidence": float(max(proba)),
        "path": steps,
        "leaf_samples": leaf_samples,
    }
