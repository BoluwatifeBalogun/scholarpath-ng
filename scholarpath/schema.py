"""Dataset schema, feature definitions and input validation.

One definition of every attribute is used by the dataset checks, the
preprocessing pipeline, the prediction form and the validator, so the
web form can never drift from what the model was trained on.
"""
from .grading import GRADING, INSTITUTION_TYPES

RESOURCE_LEVELS = ["Low", "Medium", "High"]

# Model input features, in training order.
FEATURES = [
    {"name": "institution_type", "label": "Institution type", "kind": "nominal",
     "choices": INSTITUTION_TYPES, "group": "programme",
     "help": "Decides the grade-point scale and the class names used."},
    {"name": "year_of_study", "label": "Year of study", "kind": "int",
     "min": 1, "max": 4, "unit": "", "group": "programme",
     "help": "1 for first year. Colleges of Education run to year 3."},
    {"name": "utme_score", "label": "UTME score", "kind": "int",
     "min": 100, "max": 400, "unit": "", "group": "record",
     "help": "JAMB UTME score used for admission, out of 400."},
    {"name": "previous_gpa", "label": "Previous GPA", "kind": "float",
     "min": 0.0, "max": 5.0, "unit": "", "group": "record",
     "help": "GPA from the last completed semester, on your institution's scale."},
    {"name": "ca_score", "label": "Continuous assessment", "kind": "float",
     "min": 0, "max": 30, "unit": "/30", "group": "record",
     "help": "Average continuous assessment mark this semester, out of 30."},
    {"name": "test_score", "label": "Test average", "kind": "float",
     "min": 0, "max": 100, "unit": "%", "group": "record",
     "help": "Average of class tests this semester, as a percentage."},
    {"name": "carryovers", "label": "Outstanding carryovers", "kind": "int",
     "min": 0, "max": 12, "unit": "", "group": "record",
     "help": "Number of failed courses still to be retaken."},
    {"name": "attendance_rate", "label": "Lecture attendance", "kind": "float",
     "min": 0, "max": 100, "unit": "%", "group": "habits",
     "help": "Share of lectures attended this semester."},
    {"name": "study_hours", "label": "Private study", "kind": "float",
     "min": 0, "max": 60, "unit": "hrs/week", "group": "habits",
     "help": "Hours of study outside lectures in a typical week."},
    {"name": "assignment_completion", "label": "Assignments submitted", "kind": "float",
     "min": 0, "max": 100, "unit": "%", "group": "habits",
     "help": "Share of assignments submitted on time."},
    {"name": "library_hours", "label": "Library use", "kind": "float",
     "min": 0, "max": 40, "unit": "hrs/week", "group": "habits",
     "help": "Hours spent in the library or e-library in a typical week."},
    {"name": "learning_resources", "label": "Access to learning resources", "kind": "ordinal",
     "choices": RESOURCE_LEVELS, "group": "habits",
     "help": "Textbooks, a computer, library and laboratory access."},
]

FEATURE_NAMES = [f["name"] for f in FEATURES]
FEATURE_BY_NAME = {f["name"]: f for f in FEATURES}
NUMERIC_FEATURES = [f["name"] for f in FEATURES if f["kind"] in ("int", "float")]
ORDINAL_FEATURES = [f["name"] for f in FEATURES if f["kind"] == "ordinal"]
NOMINAL_FEATURES = [f["name"] for f in FEATURES if f["kind"] == "nominal"]

TARGET_SOURCE = "final_cgpa"      # column the target is derived from
TARGET = "tier"                   # derived target used for training

# Columns present in the dataset but deliberately kept out of the model.
EXCLUDED_COLUMNS = {
    "student_id": "An identifier carries no academic information.",
    "gender": "Not an academic factor; excluded to avoid biased predictions.",
    "state": "Location is not an academic factor; kept only to describe coverage.",
    "zone": "Location is not an academic factor; kept only to describe coverage.",
    "final_cgpa": "This is the outcome the target is derived from, so using it "
                  "as an input would leak the answer.",
}

REQUIRED_COLUMNS = FEATURE_NAMES + [TARGET_SOURCE]

GROUPS = [
    ("programme", "Programme", "Where the student is studying."),
    ("record", "Academic record", "Scores already on record."),
    ("habits", "Study habits and resources", "How the semester is going."),
]


def gpa_max(institution_type):
    return GRADING[institution_type]["scale"] if institution_type in GRADING else 5.0


def year_max(institution_type):
    return GRADING[institution_type]["years"] if institution_type in GRADING else 4


def validate_input(raw: dict):
    """Validate one submitted student record.

    Returns (clean, errors). `clean` holds typed values in feature order
    and is only complete when `errors` is empty. `errors` maps a field
    name to a message that says what is wrong and how to fix it.
    """
    clean, errors = {}, {}
    itype = (raw.get("institution_type") or "").strip()

    for f in FEATURES:
        name, label = f["name"], f["label"]
        value = raw.get(name)
        value = value.strip() if isinstance(value, str) else value
        if value is None or value == "":
            errors[name] = f"Enter {label.lower()}."
            continue

        if f["kind"] in ("nominal", "ordinal"):
            if value not in f["choices"]:
                errors[name] = f"Choose one of: {', '.join(f['choices'])}."
            else:
                clean[name] = value
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[name] = f"{label} must be a number."
            continue
        if number != number or number in (float("inf"), float("-inf")):
            errors[name] = f"{label} must be a number."
            continue

        lo, hi = f["min"], f["max"]
        if name == "previous_gpa" and itype in GRADING:
            hi = gpa_max(itype)
        if name == "year_of_study" and itype in GRADING:
            hi = year_max(itype)

        if f["kind"] == "int":
            if number != int(number):
                errors[name] = f"{label} must be a whole number."
                continue
            number = int(number)

        if number < lo or number > hi:
            if name == "previous_gpa" and itype in GRADING:
                errors[name] = (f"Previous GPA must be between 0 and {hi:g} "
                                f"for a {itype}.")
            elif name == "year_of_study" and itype in GRADING:
                errors[name] = f"A {itype} programme runs from year 1 to {hi}."
            else:
                errors[name] = f"{label} must be between {lo:g} and {hi:g}."
            continue
        clean[name] = number

    return clean, errors
