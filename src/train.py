"""
Train a US-centric insurance PREMIUM CATEGORY classifier.

The raw dataset (US "Medical Cost Personal" dataset) has a continuous USD
`charges` column. Since this project predicts a *category* (not an exact
amount), we bucket `charges` into 4 business bands using quartiles:

    Low  <  Medium  <  High  <  Very High

The trained artifact is a single scikit-learn Pipeline (preprocessing +
classifier), so the FastAPI service only has to call `.predict(...)`.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "insurance.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

# Feature definitions (must stay in sync with app/main.py)
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "premium_category"

CATEGORY_LABELS = ["Low", "Medium", "High", "Very High"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    expected = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df


def make_categories(df: pd.DataFrame):
    """Bucket continuous USD `charges` into 4 quartile-based categories.

    Returns the labeled dataframe and the dollar bin edges (so we can document
    what each category means in $).
    """
    df = df.copy()
    df[TARGET], bins = pd.qcut(
        df["charges"], q=4, labels=CATEGORY_LABELS, retbins=True
    )
    df[TARGET] = df[TARGET].astype(str)
    return df, bins


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def main() -> None:
    print(">> Loading data ...")
    df = load_data()
    df, bins = make_categories(df)

    print("\n>> Premium category meaning (USD charges -> category):")
    for i, label in enumerate(CATEGORY_LABELS):
        print(f"   {label:<10} ${bins[i]:>10,.0f}  ..  ${bins[i + 1]:>10,.0f}")

    print("\n>> Class balance:")
    print(df[TARGET].value_counts().reindex(CATEGORY_LABELS).to_string())

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n>> Training RandomForest classifier ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    print("\n>> Evaluation on held-out test set:")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, labels=CATEGORY_LABELS))
    print(">> Confusion matrix (rows=true, cols=pred), order:", CATEGORY_LABELS)
    print(confusion_matrix(y_test, y_pred, labels=CATEGORY_LABELS))

    # Persist the pipeline plus metadata the API needs at serve time.
    artifact = {
        "pipeline": pipe,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": CATEGORY_LABELS,
        "bin_edges_usd": [float(b) for b in bins],
        "sklearn_classes": pipe.named_steps["model"].classes_.tolist(),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n>> Saved model artifact -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
