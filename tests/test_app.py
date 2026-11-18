"""Integration and system tests (report Section 3.8 and Table 3.1)."""
import io

from conftest import VALID, csrf, login


def form(client, **changes):
    data = {k: str(v) for k, v in VALID.items()}
    data.update(changes)
    data["csrf_token"] = csrf(client, "/predict")
    return data


# User interface -------------------------------------------------------------

def test_public_pages_load(client):
    for path in ("/", "/login", "/register"):
        assert client.get(path).status_code == 200
    home = client.get("/").get_data(as_text=True)
    assert "simulated" in home and "Decision Tree" in home
    assert client.get("/healthz").get_json() == {"status": "ok", "model_active": True}


def test_protected_pages_need_sign_in(client):
    for path in ("/predict", "/history", "/admin/"):
        r = client.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_wrong_password_is_refused(client):
    r = login(client, "admin@scholarpath.ng", "wrong")
    assert r.status_code == 401 and "do not match" in r.get_data(as_text=True)


def test_post_without_csrf_token_is_refused(client):
    assert client.post("/login", data={"email": "a@b.c", "password": "x"}).status_code == 400


def test_ordinary_user_cannot_open_admin(user_client):
    assert user_client.get("/admin/").status_code == 403
    assert user_client.get("/admin/models").status_code == 403


# Input validation -----------------------------------------------------------

def test_incomplete_form_is_rejected_with_messages(user_client):
    r = user_client.post("/predict", data={"csrf_token": csrf(user_client, "/predict")})
    html = r.get_data(as_text=True)
    assert r.status_code == 422
    assert "to get a prediction" in html and "Enter utme score." in html


def test_out_of_range_value_is_rejected(user_client):
    r = user_client.post("/predict", data=form(user_client, attendance_rate="130"))
    assert r.status_code == 422
    assert "Lecture attendance must be between 0 and 100." in r.get_data(as_text=True)


# Data submission, model integration, prediction, result display -------------

def test_full_prediction_flow(user_client):
    r = user_client.post("/predict", data=form(user_client, reference="YCT/ND/001"))
    assert r.status_code == 302 and "/predictions/" in r.headers["Location"]
    page = user_client.get(r.headers["Location"])
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "YCT/ND/001" in html and "On course for" in html and "Polytechnic" in html
    assert any(label in html for label in
               ("Distinction", "Upper Credit", "Lower Credit", "Pass", "Fail"))
    assert "The questions the tree asked" in html and "Recommended next steps" in html
    assert "YCT/ND/001" in user_client.get("/history").get_data(as_text=True)

    pdf = user_client.get(r.headers["Location"] + "/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:5] == b"%PDF-" and len(pdf.data) > 2000


def test_class_names_follow_institution_type(user_client):
    r = user_client.post("/predict", data=form(
        user_client, institution_type="University", previous_gpa="4.7", attendance_rate="97",
        ca_score="28", test_score="88", carryovers="0", study_hours="24",
        assignment_completion="99", learning_resources="High", library_hours="9",
        utme_score="300"))
    html = user_client.get(r.headers["Location"]).get_data(as_text=True)
    assert "First Class" in html and "Second Class Upper" in html
    assert "Upper Credit" not in html


def test_json_api(user_client):
    token = csrf(user_client, "/predict")
    ok = user_client.post("/api/predict", json=VALID, headers={"X-CSRF-Token": token})
    body = ok.get_json()
    assert ok.status_code == 200 and body["ok"] and body["tier"] in (1, 2, 3, 4, 5)
    assert body["label"] and body["path"]
    bad = user_client.post("/api/predict", json=dict(VALID, carryovers=-1),
                           headers={"X-CSRF-Token": token})
    assert bad.status_code == 422 and "carryovers" in bad.get_json()["errors"]


def test_users_cannot_see_each_others_predictions(app, user_client):
    r = user_client.post("/predict", data=form(user_client, reference="private"))
    other = app.test_client()
    other.post("/register", data={"csrf_token": csrf(other, "/register"), "name": "Other Person",
                                  "email": "other@example.edu.ng", "occupation": "Student",
                                  "password": "password123"})
    assert other.get(r.headers["Location"]).status_code == 404


# Administrator use cases ----------------------------------------------------

def test_admin_pages_load(admin_client):
    for path in ("/admin/", "/admin/datasets", "/admin/datasets/1", "/admin/models",
                 "/admin/models/1", "/admin/models/1/rules.txt", "/admin/datasets/template.csv"):
        assert admin_client.get(path).status_code == 200, path
    html = admin_client.get("/admin/models/1").get_data(as_text=True)
    for word in ("Confusion matrix", "Precision", "Recall", "F1", "Accuracy"):
        assert word in html


def test_upload_rejects_bad_files(admin_client):
    token = csrf(admin_client, "/admin/datasets")
    r = admin_client.post("/admin/datasets", data={
        "csrf_token": token, "name": "bad",
        "file": (io.BytesIO(b"a,b\n1,2\n"), "bad.csv")}, content_type="multipart/form-data")
    assert r.status_code == 422 and "missing these required columns" in r.get_data(as_text=True)
    r = admin_client.post("/admin/datasets", data={
        "csrf_token": token, "file": (io.BytesIO(b"x"), "notes.txt")},
        content_type="multipart/form-data")
    assert r.status_code == 422 and "Only .csv files" in r.get_data(as_text=True)


def test_upload_train_evaluate_activate(app, admin_client, raw_df):
    # upload a dataset
    data = raw_df.sample(1500, random_state=1).to_csv(index=False).encode()
    r = admin_client.post("/admin/datasets", data={
        "csrf_token": csrf(admin_client, "/admin/datasets"), "name": "Sample upload",
        "file": (io.BytesIO(data), "sample.csv")}, content_type="multipart/form-data")
    assert r.status_code == 302
    dataset_id = int(r.headers["Location"].rsplit("/", 1)[1])

    # train on it with chosen settings
    r = admin_client.post("/admin/models", data={
        "csrf_token": csrf(admin_client, "/admin/models"), "dataset_id": dataset_id,
        "criterion": "gini", "max_depth": 5, "min_samples_leaf": 25, "test_pct": 25})
    assert r.status_code == 302
    model_url = r.headers["Location"]
    html = admin_client.get(model_url).get_data(as_text=True)
    assert "Gini impurity" in html and "Put this version in use" in html

    # the old model is still the active one until the new one is activated
    from scholarpath import services
    with app.app_context():
        db = app.db()
        before = services.active_model(db).id
        r = admin_client.post(model_url + "/activate",
                              data={"csrf_token": csrf(admin_client, "/admin/models")})
        assert r.status_code == 302
        db.expire_all()
        after = services.active_model(db)
        assert after.id != before and after.params["max_depth"] == 5
        # put the original model back for the other tests
        services.activate_model(db, db.get(type(after), before))
        app.db.remove()


def test_bad_training_settings_show_a_message(admin_client):
    r = admin_client.post("/admin/models", data={
        "csrf_token": csrf(admin_client, "/admin/models"), "dataset_id": 1,
        "criterion": "gini", "max_depth": 50, "min_samples_leaf": 5, "test_pct": 20})
    assert r.status_code == 422 and "between 2 and 20" in r.get_data(as_text=True)


def test_active_model_cannot_be_deleted(admin_client):
    r = admin_client.post("/admin/models/1/delete",
                          data={"csrf_token": csrf(admin_client, "/admin/models")},
                          follow_redirects=True)
    assert "cannot be deleted" in r.get_data(as_text=True)
    assert admin_client.get("/admin/models/1").status_code == 200


def test_automatic_tuning_ignores_blank_manual_settings(admin_client):
    # the browser disables the three manual fields when tuning is ticked
    r = admin_client.post("/admin/models", data={
        "csrf_token": csrf(admin_client, "/admin/models"), "dataset_id": 1,
        "tune": "on", "test_pct": 20})
    assert r.status_code == 302
    html = admin_client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Grid search, 32 candidates, 5-fold CV" in html


def test_home_demo_endpoint(client):
    r = client.get("/api/demo?type=Polytechnic&gpa=3.6&attendance=95&ca=24")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] and body["tier"] in (1, 2, 3, 4, 5)
    assert body["label"] in ("Distinction", "Upper Credit", "Lower Credit", "Pass", "Fail")
    assert client.get("/api/demo?type=Polytechnic&gpa=9&attendance=95&ca=24").status_code == 422


def test_what_if_does_not_save(app, user_client):
    from scholarpath.db import Prediction
    with app.app_context():
        before = app.db().query(Prediction).count(); app.db.remove()
    token = csrf(user_client, "/predict")
    r = user_client.post("/api/whatif", json=dict(VALID, attendance_rate=99),
                         headers={"X-CSRF-Token": token})
    assert r.status_code == 200 and r.get_json()["label"]
    bad = user_client.post("/api/whatif", json=dict(VALID, attendance_rate=999),
                           headers={"X-CSRF-Token": token})
    assert bad.status_code == 422
    with app.app_context():
        assert app.db().query(Prediction).count() == before; app.db.remove()
