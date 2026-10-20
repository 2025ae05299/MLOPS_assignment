PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup download eda train test lint serve infer report notebook docker-build

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download:
	$(PYTHON) scripts/download_data.py

eda:
	$(PYTHON) scripts/run_eda.py

train:
	$(PYTHON) scripts/train.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src scripts tests

serve:
	$(PYTHON) -m uvicorn pulseguard.serving.app:app --app-dir src --host 0.0.0.0 --port 8000

infer:
	$(PYTHON) scripts/infer.py examples/patient_elevated.json

report:
	$(PYTHON) scripts/build_report.py

notebook:
	$(PYTHON) scripts/build_notebook.py

docker-build:
	docker build -t pulseguard-api:1.0.0 .
