"""Grading structure adopted for the study.

The model predicts one of five ordered performance tiers (1 = highest).
Each Nigerian institution type names those tiers differently and places
them on its own grade-point scale, so the label shown to a user always
follows the institution type they selected.

Tier boundaries are expressed on each institution's own CGPA scale:
  University             5-point scale (NUC classification of degrees)
  Polytechnic            4-point scale (NBTE classification of diplomas)
  College of Education   5-point scale (NCCE classification of the NCE)
"""

INSTITUTION_TYPES = ["University", "Polytechnic", "College of Education"]

GRADING = {
    "University": {
        "scale": 5.0,
        "body": "NUC",
        "years": 4,
        # (tier, label, minimum CGPA)
        "bands": [
            (1, "First Class", 4.50),
            (2, "Second Class Upper", 3.50),
            (3, "Second Class Lower", 2.40),
            (4, "Third Class", 1.50),
            (5, "Fail", 0.00),
        ],
    },
    "Polytechnic": {
        "scale": 4.0,
        "body": "NBTE",
        "years": 4,
        "bands": [
            (1, "Distinction", 3.50),
            (2, "Upper Credit", 3.00),
            (3, "Lower Credit", 2.50),
            (4, "Pass", 2.00),
            (5, "Fail", 0.00),
        ],
    },
    "College of Education": {
        "scale": 5.0,
        "body": "NCCE",
        "years": 3,
        "bands": [
            (1, "Distinction", 4.50),
            (2, "Credit", 3.50),
            (3, "Merit", 2.40),
            (4, "Pass", 1.50),
            (5, "Fail", 0.00),
        ],
    },
}

TIERS = [1, 2, 3, 4, 5]

# Plain-language meaning of each tier, independent of institution type.
TIER_MEANING = {
    1: "Excellent standing",
    2: "Strong standing",
    3: "Satisfactory standing",
    4: "Weak standing, support advised",
    5: "At risk of failing, support needed now",
}


def scale_for(institution_type: str) -> float:
    return GRADING[institution_type]["scale"]


def tier_from_cgpa(institution_type: str, cgpa: float) -> int:
    """Return the tier (1-5) a CGPA falls into for an institution type."""
    for tier, _label, minimum in GRADING[institution_type]["bands"]:
        if cgpa >= minimum:
            return tier
    return 5


def label_for(institution_type: str, tier: int) -> str:
    for t, label, _minimum in GRADING[institution_type]["bands"]:
        if t == int(tier):
            return label
    raise ValueError(f"Unknown tier {tier!r}")


def bands_table(institution_type: str):
    """Bands with their upper bound, for display."""
    g = GRADING[institution_type]
    out, upper = [], g["scale"]
    for tier, label, minimum in g["bands"]:
        out.append({"tier": tier, "label": label, "min": minimum, "max": upper,
                    "meaning": TIER_MEANING[tier]})
        upper = round(minimum - 0.01, 2)
    return out
