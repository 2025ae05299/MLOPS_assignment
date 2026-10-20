FROM python:3.9-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PULSEGUARD_MODEL_PATH=/app/models/pulseguard_pipeline.joblib

RUN adduser --disabled-password --gecos "" pulseguard
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY models ./models
USER pulseguard
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --retries=4 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "pulseguard.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
