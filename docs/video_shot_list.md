# Video shot list

Record this yourself. The repository cannot include a screen recording.

Suggested length: 4 to 6 minutes.

1. Show the repository root and `README.md`.
2. Run `make test` and point at one data test and one API test.
3. Open `reports/figures/missingness_by_site.png` and explain the Switzerland cholesterol sentinel.
4. Open MLflow (`mlflow ui --backend-store-uri mlruns`) and compare the three candidate runs.
5. Build and run the Docker image, then `curl` `/health` and `/predict` with `examples/patient_elevated.json`.
6. Open Grafana, or show `/metrics` if the compose stack is not running.
7. Show `.github/workflows/ci.yml` and the Actions run.
8. Close by reading the educational disclaimer: this API is not a diagnostic device.
