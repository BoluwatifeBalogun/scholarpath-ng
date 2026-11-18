"""Generate the simulated Nigeria-wide student academic dataset.

IMPORTANT: these are NOT real student records. No public dataset of
student academic records covering all Nigerian tertiary institutions
was available for this study, so this script simulates one. Every
value is drawn from the distributions below with a fixed random seed,
which makes the dataset fully reproducible.

The file is deliberately left "raw": it contains missing values,
inconsistent category spellings, out-of-range entries and duplicate
rows, so that the preprocessing stage in Section 3.4.4 of the report
has real work to do.

Usage:  python scripts/generate_dataset.py [rows] [output.csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scholarpath.grading import GRADING  # noqa: E402

SEED = 4117

ZONES = {
    "North Central": ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau", "FCT"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
}
# Rough relative tertiary enrolment weight per state (illustrative only).
HEAVY = {"Lagos": 3.0, "Oyo": 2.2, "Kano": 2.0, "Rivers": 1.9, "Kaduna": 1.8,
         "Enugu": 1.7, "Ogun": 1.7, "Edo": 1.6, "Anambra": 1.6, "FCT": 1.6,
         "Osun": 1.5, "Imo": 1.5, "Delta": 1.5, "Kwara": 1.4}


def generate(n=7500, seed=SEED):
    rng = np.random.default_rng(seed)

    states, zones, weights = [], [], []
    for zone, names in ZONES.items():
        for s in names:
            states.append(s)
            zones.append(zone)
            weights.append(HEAVY.get(s, 1.0))
    weights = np.array(weights) / sum(weights)
    idx = rng.choice(len(states), size=n, p=weights)

    itype = rng.choice(["University", "Polytechnic", "College of Education"],
                       size=n, p=[0.44, 0.36, 0.20])
    scale = np.array([GRADING[t]["scale"] for t in itype])
    years = np.array([GRADING[t]["years"] for t in itype])
    year = np.array([rng.integers(1, y + 1) for y in years])

    ability = rng.normal(0, 1, n)              # unobserved academic ability
    diligence = rng.normal(0, 1, n)            # unobserved effort this semester

    utme_base = np.where(itype == "University", 228,
                         np.where(itype == "Polytechnic", 198, 182))
    utme = np.clip(utme_base + 26 * ability + rng.normal(0, 20, n), 120, 372).round()

    attendance = np.clip(75 + 4 * ability + 10 * diligence + rng.normal(0, 7, n), 15, 100).round(0)
    study = np.clip(12 + 2.0 * ability + 4.5 * diligence + rng.normal(0, 3.5, n), 0, 48).round(0)
    assignments = np.clip(79 + 3 * ability + 9 * diligence + rng.normal(0, 8, n), 5, 100).round(0)

    res_latent = 0.35 * ability + rng.normal(0, 1, n)
    resources = np.select([res_latent < -0.55, res_latent < 0.75], ["Low", "Medium"], "High")
    library = np.clip(4.0 + 1.1 * ability + 2.4 * diligence + 0.8 * res_latent
                      + rng.normal(0, 2.2, n), 0, 30).round(0)
    res_idx = np.select([resources == "Low", resources == "Medium"], [0, 1], 2)

    prev_ratio = np.clip(0.615 + 0.125 * ability + 0.03 * diligence + rng.normal(0, 0.055, n), 0.12, 1.0)
    previous_gpa = (prev_ratio * scale).round(2)

    carry_rate = np.exp(-0.35 - 0.95 * ability - 0.35 * diligence)
    carryovers = np.clip(rng.poisson(carry_rate), 0, 9)
    carryovers = np.where(year == 1, np.minimum(carryovers, 2), carryovers)

    ca = np.clip(18.5 + 2.6 * ability + 2.2 * diligence + rng.normal(0, 2.4, n), 2, 30).round(1)
    test = np.clip(56 + 9 * ability + 6 * diligence + rng.normal(0, 8, n), 5, 100).round(0)

    # Final cumulative standing: mostly carried by the previous record,
    # moved by this semester's effort, resources and outstanding courses.
    final_ratio = (
        0.52 * prev_ratio
        + 0.48 * (0.615 + 0.105 * ability + 0.055 * diligence)
        + 0.0011 * (attendance - 75)
        + 0.0016 * (study - 12)
        + 0.0030 * (ca - 18.5)
        + 0.0008 * (test - 56)
        + 0.010 * (res_idx - 1)
        + 0.0022 * (library - 4)
        - 0.014 * carryovers
        + rng.normal(0, 0.030, n)
    )
    final_ratio = np.clip(final_ratio, 0.05, 1.0)
    final_cgpa = (final_ratio * scale).round(2)

    df = pd.DataFrame({
        "student_id": [f"SP{i:06d}" for i in range(1, n + 1)],
        "gender": rng.choice(["Female", "Male"], size=n, p=[0.46, 0.54]),
        "state": [states[i] for i in idx],
        "zone": [zones[i] for i in idx],
        "institution_type": itype,
        "year_of_study": year,
        "utme_score": utme.astype(int),
        "previous_gpa": previous_gpa,
        "ca_score": ca,
        "test_score": test,
        "carryovers": carryovers,
        "attendance_rate": attendance,
        "study_hours": study,
        "assignment_completion": assignments,
        "library_hours": library,
        "learning_resources": resources,
        "final_cgpa": final_cgpa,
    })
    return dirty(df, rng)


def dirty(df, rng):
    """Introduce the data-quality problems real record sheets contain."""
    n = len(df)
    df = df.astype({"attendance_rate": "float", "study_hours": "float",
                    "test_score": "float", "assignment_completion": "float", "library_hours": "float",
                    "utme_score": "float"})

    # 1. Missing values
    for col, rate in [("attendance_rate", 0.018), ("study_hours", 0.022),
                      ("ca_score", 0.012), ("test_score", 0.010),
                      ("utme_score", 0.015), ("assignment_completion", 0.012),
                      ("learning_resources", 0.014), ("library_hours", 0.016)]:
        df.loc[rng.random(n) < rate, col] = np.nan
    df.loc[rng.random(n) < 0.004, "final_cgpa"] = np.nan

    # 2. Inconsistent category spellings
    variants = {
        "University": ["university", "UNIVERSITY", " University ", "Uni"],
        "Polytechnic": ["polytechnic", "POLYTECHNIC", "Poly", "Polytechnic "],
        "College of Education": ["college of education", "COE", "College Of Education"],
    }
    mask = rng.random(n) < 0.03
    df.loc[mask, "institution_type"] = [
        rng.choice(variants[v]) for v in df.loc[mask, "institution_type"]]
    for col in ("learning_resources",):
        mask = (rng.random(n) < 0.02) & df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].str.lower()

    # 3. Out-of-range entries (typing errors)
    m = rng.random(n) < 0.004
    df.loc[m, "attendance_rate"] = rng.integers(101, 160, m.sum())
    m = rng.random(n) < 0.003
    df.loc[m, "study_hours"] = -rng.integers(1, 9, m.sum())
    m = rng.random(n) < 0.003
    df.loc[m, "ca_score"] = rng.integers(31, 60, m.sum())
    m = rng.random(n) < 0.002
    df.loc[m, "previous_gpa"] = rng.integers(6, 10, m.sum())

    # 4. Duplicate rows
    dupes = df.sample(frac=0.012, random_state=int(rng.integers(1e6)))
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 7500
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parents[1] / "data" / "tertiary_records_ng_simulated.csv"
    frame = generate(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame)} rows to {out}")
