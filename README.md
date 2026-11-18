# ScholarPath NG

A web system that predicts a tertiary student's class of result with a
Decision Tree, for Nigerian universities, polytechnics and colleges of
education.

Implementation of the seminar project *Development of a Data Driven Predictive
System for Student Academic Performance Using Decision Tree Algorithm*
(Department of Computer Science, Yaba College of Technology).

The system supports decisions. It does not replace a lecturer's assessment or
an institution's grading.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # then open http://localhost:5000
```

The first start builds the database, loads the bundled dataset, trains the
tree and creates the administrator:

| | |
|---|---|
| Email | `admin@scholarpath.ng` |
| Password | `Scholar@2026` |

Set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in the environment before hosting.
Students, lecturers and academic officers register themselves.

```bash
python -m pytest tests -q          # 56 tests
python scripts/train.py --tune     # train and print the evaluation
python scripts/generate_dataset.py # rebuild the bundled dataset
```

## Features

**Student, lecturer or academic officer**
- Live demo on the home page that calls the trained model
- One-page record form with instant checks and example records
- Predicted class of result in the institution's own terms
- The exact tests the tree applied, in order
- A what-if panel: change attendance, study time, CA, tests or carryovers and
  the model answers again without saving
- Recommended next steps, saved history and a PDF per prediction

**System administrator**
- Upload a CSV dataset; it is inspected and cleaned, and the log is kept
- Train with chosen settings or a cross-validated grid search
- Accuracy, precision, recall, F1, confusion matrix, feature importance and
  a drawing of the learned tree for every version
- Choose which version answers predictions

## Map of the code

| Report section | File |
|---|---|
| 3.3 User interface | `scholarpath/templates/`, `scholarpath/static/` |
| 3.3 Backend | `scholarpath/__init__.py`, `scholarpath/routes/` |
| 3.4.2 Inputs and output | `scholarpath/schema.py`, `scholarpath/grading.py` |
| 3.4.4 Preprocessing | `scholarpath/preprocessing.py` |
| 3.5.1 Training, evaluation, saving | `scholarpath/modeling.py` |
| 3.5.2 Use cases | `routes/predict.py` (user), `routes/admin.py` (administrator) |
| 3.6 Evaluation | Models, then any version, in the admin area |
| 3.8 Testing | `tests/` |
| Storage | `scholarpath/db.py` |

## About the data

`data/tertiary_records_ng_simulated.csv` is **simulated**. No public dataset
of student records covering all Nigerian tertiary institutions was available,
so `scripts/generate_dataset.py` generates one (7,590 raw rows, 36 states and
the FCT, fixed seed) and leaves in the faults real record sheets have: blanks,
inconsistent spellings, impossible values and repeated rows.

The evaluation figures show the pipeline works. They say nothing about
accuracy on real students. To use real records, upload a CSV with the same
columns from the Datasets page (a template is offered there), then train and
put the new version in use.

## Grading structure

| Tier | University (5.0) | Polytechnic (4.0) | College of Education (5.0) |
|---|---|---|---|
| 1 | First Class, 4.50+ | Distinction, 3.50+ | Distinction, 4.50+ |
| 2 | Second Class Upper, 3.50+ | Upper Credit, 3.00+ | Credit, 3.50+ |
| 3 | Second Class Lower, 2.40+ | Lower Credit, 2.50+ | Merit, 2.40+ |
| 4 | Third Class, 1.50+ | Pass, 2.00+ | Pass, 1.50+ |
| 5 | Fail | Fail | Fail |

Commonly used NUC, NBTE and NCCE bands. Check them against the handbook of
the institution being studied; they are all in `scholarpath/grading.py`.

## Hosting

`Procfile` and `render.yaml` are included. Set `SECRET_KEY` and
`ADMIN_PASSWORD`. On a host that clears local files on restart, also set
`DATABASE_URL` to a PostgreSQL database; datasets and trained models live in
the database too.

The product name is `APP_NAME` in `scholarpath/config.py`. Design notes are in
`docs/PLAN.md` and Chapter 4 figures in `docs/RESULTS.md`.
