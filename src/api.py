"""FastAPI деплой модели IMU → 3D-скелет.

Запуск:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /            — html-страница с описанием
    GET  /health      — статус сервиса
    GET  /schema      — описание входного/выходного формата
    POST /predict     — предсказание скелета по сырым IMU + калибровкам

Модель: Linear Regression (победитель сравнения, MPJPE 0.11 на тесте).
Веса/scaler загружаются из models/linear.joblib + data/processed/scaler.npz.
Если линейной модели нет — она дообучается на лету (≈1 сек).
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src import preprocessing as pp
from src.skeleton import EDGES, JOINTS

MODEL_PATH = os.environ.get("MODEL_PATH", "models/linear.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "data/processed/scaler.npz")


# ----------------------------- I/O схемы -----------------------------


class SensorReading(BaseModel):
    quat: list[float] = Field(..., description="Ориентация IMU, кватернион [w, x, y, z]", min_length=4, max_length=4)
    accel: list[float] = Field(..., min_length=3, max_length=3)
    gyro: list[float] = Field(..., min_length=3, max_length=3)
    mag: list[float] = Field(..., min_length=3, max_length=3)


class Calibration(BaseModel):
    ref: list[float] = Field(..., description="calib_ref, кватернион [x, y, z, w]", min_length=4, max_length=4)
    bone: list[float] = Field(..., description="calib_bone, кватернион [x, y, z, w]", min_length=4, max_length=4)


class PredictRequest(BaseModel):
    frames: list[dict[str, SensorReading]] = Field(
        ..., description="Список кадров. Каждый кадр: словарь {sensor_name → reading}. "
                         "Имена сенсоров: Head, Sternum, Pelvis, L_UpArm, R_UpArm, L_LowArm, "
                         "R_LowArm, L_UpLeg, R_UpLeg, L_LowLeg, R_LowLeg, L_Foot, R_Foot.")
    calibration: dict[str, Calibration] = Field(
        ..., description="Калибровка для каждого сенсора (одна на всю запись).")
    hips: list[list[float]] | None = Field(
        default=None,
        description="Опционально: позиция Hips (N,3) в системе Vicon. "
                    "Если не передана — заполняется нулями, и предсказание возвращается в hip-relative координатах.")


class PredictResponse(BaseModel):
    joints: list[str]
    edges: list[list[int]]
    positions: list[list[list[float]]] = Field(..., description="(T, 14, 3) предсказанные позиции суставов")
    units: str = "BVH (≈ см)"
    n_frames: int


# ----------------------------- Загрузка модели -----------------------------


class ModelBundle:
    model: Any = None
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None

    @classmethod
    def load(cls) -> None:
        if not os.path.exists(SCALER_PATH):
            raise RuntimeError(f"Не найден scaler: {SCALER_PATH}. Запустите src.preprocessing.")
        s = np.load(SCALER_PATH, allow_pickle=True)
        cls.mu, cls.sigma = s["mu"], s["sigma"]

        if os.path.exists(MODEL_PATH):
            cls.model = joblib.load(MODEL_PATH)
            return

        # fallback: дообучаем linear на train.npz
        train_path = os.path.join(os.path.dirname(SCALER_PATH), "train.npz")
        if not os.path.exists(train_path):
            raise RuntimeError("Нет сохранённой модели и нет train.npz для обучения на лету.")
        from sklearn.linear_model import LinearRegression
        train = dict(np.load(train_path, allow_pickle=True))
        cls.model = LinearRegression(n_jobs=-1).fit(train["X_norm"], train["Y"])
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(cls.model, MODEL_PATH)


# ----------------------------- Feature engineering -----------------------------


def build_features(req: PredictRequest) -> np.ndarray:
    """Повторяет логику src.preprocessing.build_take для пользовательских данных."""
    n = len(req.frames)
    if n == 0:
        raise HTTPException(status_code=400, detail="frames пуст")

    per_sensor_feats = []
    for s in pp.SENSOR_ORDER:
        if s not in req.calibration:
            raise HTTPException(status_code=400, detail=f"calibration['{s}'] отсутствует")
        quat = np.zeros((n, 4), dtype=np.float32)
        accel = np.zeros((n, 3), dtype=np.float32)
        gyro = np.zeros((n, 3), dtype=np.float32)
        mag = np.zeros((n, 3), dtype=np.float32)
        for t, frame in enumerate(req.frames):
            if s not in frame:
                raise HTTPException(status_code=400, detail=f"frame[{t}]['{s}'] отсутствует")
            r = frame[s]
            quat[t] = r.quat
            accel[t] = r.accel
            gyro[t] = r.gyro
            mag[t] = r.mag
        calib = req.calibration[s]
        cal_quat = pp.calibrate_orientation(
            quat, np.asarray(calib.ref, dtype=np.float32), np.asarray(calib.bone, dtype=np.float32))
        feats = np.concatenate([
            cal_quat, accel, gyro, mag,
            pp.finite_diff(cal_quat), pp.finite_diff(accel), pp.finite_diff(gyro),
        ], axis=1)
        per_sensor_feats.append(feats)

    hips = (np.asarray(req.hips, dtype=np.float32) if req.hips is not None
            else np.zeros((n, 3), dtype=np.float32))
    if hips.shape != (n, 3):
        raise HTTPException(status_code=400, detail=f"hips должен иметь форму ({n}, 3)")

    X = np.concatenate(per_sensor_feats + [hips, pp.finite_diff(hips)], axis=1)
    return X


# ----------------------------- App -----------------------------


app = FastAPI(
    title="IMU → 3D Skeleton",
    description="Регрессия 3D-позиций 14 суставов верхней части тела по 13 IMU-сенсорам.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    ModelBundle.load()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": ModelBundle.model is not None,
        "n_features": int(ModelBundle.mu.shape[0]) if ModelBundle.mu is not None else None,
        "n_joints": len(JOINTS),
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "sensors": pp.SENSOR_ORDER,
        "sensor_fields": {
            "quat": "Ориентация IMU, кватернион [w, x, y, z]",
            "accel": "Акселерометр [x, y, z], м/с²",
            "gyro": "Гироскоп [x, y, z], рад/с",
            "mag": "Магнитометр [x, y, z]",
        },
        "calibration_fields": {
            "ref": "calib_ref квaт [x, y, z, w] — ориентация IMU в reference-позе",
            "bone": "calib_bone квaт [x, y, z, w] — поворот кости относительно IMU",
        },
        "joints": JOINTS,
        "edges": EDGES,
        "hips_optional": True,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if ModelBundle.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    X = build_features(req)
    X_norm = (X - ModelBundle.mu) / ModelBundle.sigma
    Y = ModelBundle.model.predict(X_norm)
    Y = Y.reshape(-1, 14, 3)
    return PredictResponse(
        joints=JOINTS,
        edges=[list(e) for e in EDGES],
        positions=Y.tolist(),
        n_frames=int(Y.shape[0]),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html><head><title>IMU → 3D Skeleton API</title>
    <style>
      body { font-family: -apple-system, system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
      code { background: #f3f3f3; padding: 2px 6px; border-radius: 3px; }
      pre { background: #f7f7f7; padding: 12px; border-radius: 6px; overflow: auto; }
      a { color: #0a66c2; }
    </style></head><body>
    <h1>IMU → 3D Skeleton</h1>
    <p>FastAPI-сервис регрессии 3D-позиций 14 суставов верхней части тела
       по 13 IMU-сенсорам. Модель: Linear Regression на калиброванных кватернионах
       (test MPJPE = 0.11 BVH ≈ см).</p>
    <h2>Endpoints</h2>
    <ul>
      <li><code>GET /schema</code> — описание входного формата</li>
      <li><code>POST /predict</code> — получить позиции суставов</li>
      <li><code>GET /health</code> — статус</li>
      <li><a href="/docs">/docs</a> — Swagger UI</li>
    </ul>
    <h2>Пример</h2>
    <pre>python -m src.client_example --frame 100 --out demo.png</pre>
    </body></html>
    """
