"""Train and evaluate the Decision Tree from the command line.

    python scripts/train.py            # default settings
    python scripts/train.py --tune     # 5-fold grid search
    python scripts/train.py --data path/to/records.csv --save model.joblib
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scholarpath import config, modeling  # noqa: E402
from scholarpath.services import read_csv_bytes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(config.BUNDLED_DATASET))
parser.add_argument("--tune", action="store_true")
parser.add_argument("--save")
args = parser.parse_args()

result = modeling.train(read_csv_bytes(Path(args.data).read_bytes()), tune=args.tune)
m, c = result["metrics"], result["cleaning"]
print(f"Rows: {c['rows_in']} raw, {c['rows_out']} usable "
      f"({c['duplicates_removed']} duplicates removed, "
      f"{c['invalid_values_set_missing']} invalid values, "
      f"{c['missing_to_impute_total']} missing values imputed)")
print(f"Settings: {result['params']}")
if result["tuning"]:
    print(f"Grid search: {result['tuning']}")
print(f"Split: {result['split']['train']} train / {result['split']['test']} test")
print(f"Tree: depth {result['tree']['depth']}, {result['tree']['leaves']} leaves")
print(f"\nAccuracy  {m['accuracy']:.4f}   (training accuracy {result['train_accuracy']:.4f})")
print(f"Macro     precision {m['macro']['precision']:.4f}  recall {m['macro']['recall']:.4f}  "
      f"f1 {m['macro']['f1']:.4f}")
print(f"Weighted  precision {m['weighted']['precision']:.4f}  recall {m['weighted']['recall']:.4f}  "
      f"f1 {m['weighted']['f1']:.4f}\n")
print("Tier  precision  recall  f1      support")
for row in m["per_class"]:
    print(f"{row['tier']:<5} {row['precision']:<10.4f} {row['recall']:<7.4f} "
          f"{row['f1']:<7.4f} {row['support']}")
print("\nConfusion matrix (rows actual, columns predicted, tiers 1 to 5)")
for row in m["confusion_matrix"]:
    print("  " + " ".join(f"{n:5d}" for n in row))
if args.save:
    Path(args.save).write_bytes(modeling.dumps(result["pipeline"]))
    print(f"\nSaved pipeline to {args.save}")
