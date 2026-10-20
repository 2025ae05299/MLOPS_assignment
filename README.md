# PulseGuard

Educational MLOps project for the UCI Heart Disease collection. It trains a binary risk-marker classifier, tracks the comparison in MLflow, and serves the frozen scikit-learn pipeline from FastAPI.

This is not a medical device. Do not use the API to diagnose, screen, or treat a person.

Repository: https://github.com/2025ae05299/MLOPS_assignment

## What is different here

- All four processed UCI sites are used, not only the 303-row Cleveland file.
- Cholesterol values of 0, mostly from Switzerland, are treated as missing measurements.
- Vessel count and thalassemia missingness are modeled explicitly because they are site-dependent.
- The selected model maximizes `0.70 * ROC-AUC + 0.30 * recall`, then a separate F1.5 threshold is frozen before the holdout split is touched.
- The random seed is 5299, from the repository owner id.
- Prediction logs store a payload hash, not the raw clinical JSON.

## Layout

| Path | Purpose |
| --- | --- |
| `scripts/download_data.py` | Download and clean the UCI files |
| `scripts/run_eda.py` | Write EDA figures and `reports/eda_summary.json` |
| `scripts/train.py` | Compare models, log MLflow, save the serving bundle |
| `scripts/infer.py` | Score one JSON file offline |
| `notebooks/01_eda_and_modeling.ipynb` | Narrative notebook over the same code |
| `src/pulseguard/` | Library, API, metrics |
| `tests/` | Pytest suite |
| `.github/workflows/ci.yml` | Lint, test, train, Docker probe |
| `Dockerfile`, `docker-compose.yml` | API plus Prometheus and Grafana |
| `k8s/`, `deploy/helm/pulseguard/` | Deployment manifests |
| `reports/PulseGuard_MLOps_Report.pdf` | Course report |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
python scripts/run_eda.py
python scripts/train.py
pytest
ruff check src scripts tests
```

Or `make setup && make download && make eda && make train && make test`.

## Serve

```bash
uvicorn pulseguard.serving.app:app --app-dir src --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/patient_elevated.json
```

Docker:

```bash
docker build -t pulseguard-api:1.0.0 .
docker run --rm -p 8000:8000 pulseguard-api:1.0.0
```

Monitoring stack:

```bash
docker compose up --build
```

Grafana is on http://127.0.0.1:3000 with user `admin` and local password `pulseguard-local`. MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns
```

## Kubernetes

```bash
docker build -t pulseguard-api:1.0.0 .
kubectl apply -k k8s/
kubectl port-forward -n pulseguard svc/pulseguard-api 8000:80
```

Details and the cloud gap are in `docs/deployment.md`. There is no public URL until an operator with a cloud account pushes the image.

## Human deliverables

- Walkthrough video: `docs/video_shot_list.md`
- Screenshots: `screenshots/README.md`

## Data citation

Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
