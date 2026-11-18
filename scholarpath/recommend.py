"""Improvement recommendations shown with every prediction.

These are plain rules applied to the values the user entered. They are
NOT produced by the Decision Tree; they turn the entered record into
practical next steps, and put first the factors the tree actually
tested on the way to its prediction.
"""
from .grading import GRADING

RULES = [
    # (feature, test, title, advice)
    ("attendance_rate", lambda r: r["attendance_rate"] < 75,
     "Raise lecture attendance",
     lambda r: f"Attendance is {r['attendance_rate']:g}%. Most institutions require 75% "
               "to sit examinations. Plan to attend every remaining lecture and "
               "collect notes for the ones already missed."),
    ("carryovers", lambda r: r["carryovers"] >= 1,
     "Clear outstanding carryovers",
     lambda r: f"There {'is' if r['carryovers'] == 1 else 'are'} {r['carryovers']} "
               f"carryover course{'s' if r['carryovers'] != 1 else ''} on record. Register "
               "them early, get past questions, and give them fixed slots in the weekly timetable."),
    ("study_hours", lambda r: r["study_hours"] < 10,
     "Increase private study time",
     lambda r: f"Private study is {r['study_hours']:g} hours a week. Build up towards 12 to 15 "
               "hours in short daily sessions rather than one long session before tests."),
    ("ca_score", lambda r: r["ca_score"] < 15,
     "Recover continuous assessment marks",
     lambda r: f"Continuous assessment is {r['ca_score']:g} out of 30, below half. Ask the "
               "lecturer which assessments remain and whether any can be made up."),
    ("test_score", lambda r: r["test_score"] < 50,
     "Work on test performance",
     lambda r: f"The test average is {r['test_score']:g}%. Review marked scripts with a "
               "lecturer or course adviser to find the topics losing the most marks."),
    ("assignment_completion", lambda r: r["assignment_completion"] < 80,
     "Submit every assignment on time",
     lambda r: f"Only {r['assignment_completion']:g}% of assignments were submitted on time. "
               "Late or missing work costs marks that are otherwise easy to keep."),
    ("previous_gpa", lambda r: r["previous_gpa"] / GRADING[r["institution_type"]]["scale"] < 0.5,
     "See your course adviser about last semester",
     lambda r: f"The previous GPA of {r['previous_gpa']:g} is below the midpoint of the "
               f"{GRADING[r['institution_type']]['scale']:g}-point scale. Book a session with "
               "the course adviser to plan course load for the coming semester."),
    ("learning_resources", lambda r: r["learning_resources"] == "Low",
     "Get better access to learning materials",
     lambda r: "Access to learning resources is low. Use the institution library and "
               "departmental laboratory hours, and share textbooks through a study group."),
    ("library_hours", lambda r: r["library_hours"] < 2,
     "Make the library part of the week",
     lambda r: f"Library use is {r['library_hours']:g} hour{'s' if r['library_hours'] != 1 else ''} "
               "a week. Two or three fixed "
               "sessions in the library or e-library give quiet time with the "
               "recommended texts."),
]


def recommend(record: dict, tier: int, path_features=()):
    """Return a list of {title, advice, on_path} ordered by relevance."""
    found = []
    for feature, test, title, advice in RULES:
        if test(record):
            found.append({"feature": feature, "title": title, "advice": advice(record),
                          "on_path": feature in path_features})
    found.sort(key=lambda d: not d["on_path"])

    if not found:
        if tier <= 2:
            found.append({"feature": None, "on_path": False,
                          "title": "Keep the current routine",
                          "advice": "None of the entered values is in a range that calls for "
                                    "attention. Keep attendance, study time and assignment "
                                    "habits where they are."})
        else:
            found.append({"feature": None, "on_path": False,
                          "title": "Talk to a course adviser",
                          "advice": "No single entered value stands out, yet the predicted "
                                    "class is below a strong standing. A course adviser can "
                                    "look at factors this system does not record."})
    if tier >= 4:
        found.append({"feature": None, "on_path": False,
                      "title": "Ask for academic support now",
                      "advice": "This prediction is an early warning. Take it to a lecturer, "
                                "course adviser or the student affairs unit while there is "
                                "still time in the semester to act."})
    return found
