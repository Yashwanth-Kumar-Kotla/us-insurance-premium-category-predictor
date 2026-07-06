"""
Quick command-line sanity check for the trained model (no server needed).

Usage:
    python src/predict.py
"""

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

SAMPLES = [
    {"age": 19, "sex": "female", "bmi": 27.9, "children": 0,
     "smoker": "yes", "region": "southwest"},
    {"age": 45, "sex": "male", "bmi": 24.1, "children": 2,
     "smoker": "no", "region": "northeast"},
    {"age": 60, "sex": "male", "bmi": 35.0, "children": 0,
     "smoker": "yes", "region": "southeast"},
    {"age": 25, "sex": "female", "bmi": 21.0, "children": 0,
     "smoker": "no", "region": "northwest"},
]


def main() -> None:
    art = joblib.load(MODEL_PATH)
    pipe = art["pipeline"]
    classes = pipe.named_steps["model"].classes_

    df = pd.DataFrame(SAMPLES)[art["features"]]
    preds = pipe.predict(df)
    probas = pipe.predict_proba(df)

    for sample, pred, proba in zip(SAMPLES, preds, probas):
        prob_str = ", ".join(
            f"{c}={p:.2f}" for c, p in zip(classes, proba)
        )
        print(f"{sample}\n  -> {pred}   ({prob_str})\n")


if __name__ == "__main__":
    main()
