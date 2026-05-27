FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src /app/src
COPY models /app/models
COPY data/processed /app/data/processed

ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/linear.joblib \
    SCALER_PATH=/app/data/processed/scaler.npz

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
