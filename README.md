# US Insurance Premium Category Predictor

A US-centric reference build of an insurance **premium category** classifier
(Low / Medium / High / Very High), with a FastAPI service and Dockerfile —
structured so you can lift the pieces into your existing dockerized project.

This replaces an India-trained model with one trained on the standard **US
"Medical Cost Personal" dataset** (charges in USD, US regions).

---

## Why this is a retrain, not a model swap

A model is tied to its **input features**. The Indian dataset and the US
dataset have *different columns, units, and categories*, so you can't drop a
US model into an India-shaped API. What changes when going US-centric:

| | India version (typical) | US version (this project) |
|---|---|---|
| Currency | INR (₹) | USD ($) |
| Features | transplants, surgeries, allergies, cancer history… | `age, sex, bmi, children, smoker, region` |
| Region values | n/a | northeast / northwest / southeast / southwest |

So three things change: **the data, the model, and the request schema.** Your
Docker + FastAPI *structure* stays the same.

---

## How "category" is defined

The raw dataset has a continuous USD `charges` column. Since this predicts a
*band*, `train.py` buckets `charges` into 4 quartiles:

| Category | USD charges range |
|---|---|
| Low | $1,122 – $4,740 |
| Medium | $4,740 – $9,382 |
| High | $9,382 – $16,640 |
| Very High | $16,640 – $63,770 |

(These exact edges are printed during training and exposed at `GET /categories`.)
If your business already has fixed premium bands, replace `pd.qcut(...)` in
`src/train.py` with `pd.cut(..., bins=[your, dollar, thresholds])`.

---

## Project layout

```
us-insurance-premium-category-predictor/
├── data/insurance.csv      # US dataset (downloaded)
├── src/train.py            # trains pipeline -> models/model.joblib
├── src/predict.py          # CLI sanity check, no server needed
├── app/main.py             # FastAPI service (schema + endpoints)
├── models/model.joblib     # trained artifact (pipeline + metadata)
├── requirements.txt
├── Dockerfile
└── README.md
```

The model artifact is a single scikit-learn **Pipeline** (StandardScaler +
OneHotEncoder + RandomForest), so serving is just `pipeline.predict(...)` — no
separate preprocessing code to keep in sync.

---

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Train (creates models/model.joblib)
python src/train.py

# 2. Quick CLI check
python src/predict.py

# 3. Serve the API
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":60,"sex":"male","bmi":35.0,"children":0,"smoker":"yes","region":"southeast"}'
# -> {"premium_category":"Very High", "probabilities":{...}}
```

---

## Run with Docker

```bash
docker build -t us-premium-predictor .
docker run -p 8000:8000 us-premium-predictor
```

---

## Model performance

RandomForest, 80/20 stratified split: **~86% accuracy**, balanced across all
four categories. Full classification report + confusion matrix print when you
run `src/train.py`.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model-loaded check |
| GET | `/categories` | what each band means in USD |
| POST | `/predict` | returns `premium_category` + class probabilities |

---

## What to copy into YOUR project

1. **`app/main.py`** — the `ApplicantFeatures` Pydantic schema is the new US
   contract. Swap your old India schema for this.
2. **`src/train.py`** — the training + bucketing logic.
3. **`models/model.joblib`** — or retrain to regenerate it.
4. Update any **frontend form fields** and **₹ → $** labels.
5. Your existing **Dockerfile** barely changes (see this one for reference).

## Data source

US "Medical Cost Personal" dataset (a.k.a. `insurance.csv`), 1,338 records.
