import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholarpath import config, create_app  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "tertiary_records_ng_simulated.csv"

VALID = dict(institution_type="Polytechnic", year_of_study=2, utme_score=205,
             previous_gpa=2.9, ca_score=19, test_score=58, carryovers=1,
             attendance_rate=78, study_hours=11, assignment_completion=82,
             learning_resources="Medium", library_hours=4)


@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv(DATA, na_values=[""], keep_default_na=False)


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.db"
    return create_app({"TESTING": True, "DATABASE_URL": f"sqlite:///{db}",
                       "SECRET_KEY": "test"})


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def login(client, email, password):
    return client.post("/login", data={"csrf_token": csrf(client), "email": email,
                                       "password": password})


@pytest.fixture()
def user_client(app):
    c = app.test_client()
    token = csrf(c, "/register")
    r = c.post("/register", data={"csrf_token": token, "name": "Test Officer",
                                  "email": "officer@example.edu.ng", "occupation": "Academic officer",
                                  "password": "password123"})
    if r.status_code != 302:          # already registered by an earlier test
        login(c, "officer@example.edu.ng", "password123")
    return c


@pytest.fixture()
def admin_client(app):
    c = app.test_client()
    assert login(c, config.ADMIN_EMAIL, config.ADMIN_PASSWORD).status_code == 302
    return c
