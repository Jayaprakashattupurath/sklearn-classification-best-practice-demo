# scikit-learn Classification Best Practice Demo (Python + React)

Small full-stack starter for a **real-time classification use case**:
- Python API with **scikit-learn** model
- React frontend to capture input and request prediction

This example uses the Iris dataset for simplicity, but the same structure applies to production tabular classification (fraud checks, churn risk, lead scoring, quality control, etc.).

## Why this structure (best practice)

Based on scikit-learn supervised learning workflow:
- Use a **Pipeline** to combine preprocessing + model.
- Keep train/test split with **stratify** for balanced evaluation.
- Use **cross-validation** to reduce overfitting risk.
- Save model artifacts with versionable files.
- Keep serving code separate from training code.

## Project tree

- `src/backend/train.py` -> trains + evaluates model and saves artifacts
- `src/backend/api.py` -> Flask API (`/predict`)
- `src/frontend` -> React app (Vite) for interactive predictions

## Quick start

## 1) Backend (Python + scikit-learn)

```bash
cd src/backend
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python train.py
python api.py
```

Backend runs at: `http://localhost:8000`

Test health:
```bash
curl http://localhost:8000/health
```

Test prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## 2) Frontend (Node + React)

```bash
cd src/frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

## Real-time scenario guidance

If adapting to a business use case:
1. Replace Iris data with your own labeled dataset.
2. Keep all feature transforms inside a Pipeline.
3. Add validation checks in API (types, ranges, missing values).
4. Log request IDs + model version for traceability.
5. Monitor drift and retrain periodically.
6. Add CI checks for reproducible training and API tests.

## Notes

- This is intentionally lightweight (educational starter).
- For production, consider FastAPI, model registry, Docker, auth, and observability.