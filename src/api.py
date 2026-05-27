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

import io
import os
import tempfile
from typing import Any

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from src import preprocessing as pp
from src.skeleton import EDGES, JOINTS
from src.visualize import animate

MLP_PATH = os.environ.get("MLP_PATH", "models/mlp.pt")
LINEAR_PATH = os.environ.get("MODEL_PATH", "models/linear.joblib")
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
    kind: str = "none"          # "mlp" | "linear"
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None

    @classmethod
    def load(cls) -> None:
        if not os.path.exists(SCALER_PATH):
            raise RuntimeError(f"Не найден scaler: {SCALER_PATH}. Запустите src.preprocessing.")
        s = np.load(SCALER_PATH, allow_pickle=True)
        cls.mu, cls.sigma = s["mu"], s["sigma"]

        # Приоритет — MLP (лучшее качество, test MPJPE 1.83)
        if os.path.exists(MLP_PATH):
            from src.modeling import MLPRegressor
            cls.model = MLPRegressor.load(MLP_PATH, device="cpu")
            cls.kind = "mlp"
            return
        if os.path.exists(LINEAR_PATH):
            cls.model = joblib.load(LINEAR_PATH)
            cls.kind = "linear"
            return

        # fallback: дообучаем linear на train.npz
        train_path = os.path.join(os.path.dirname(SCALER_PATH), "train.npz")
        if not os.path.exists(train_path):
            raise RuntimeError("Нет сохранённой модели и нет train.npz для обучения на лету.")
        from sklearn.linear_model import LinearRegression
        train = dict(np.load(train_path, allow_pickle=True))
        cls.model = LinearRegression(n_jobs=-1).fit(train["X_norm"], train["Y"])
        cls.kind = "linear"
        os.makedirs(os.path.dirname(LINEAR_PATH), exist_ok=True)
        joblib.dump(cls.model, LINEAR_PATH)


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
        "model_kind": ModelBundle.kind,
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


def _features_from_raw_imu(raw_imu_T_S13: np.ndarray, hips: np.ndarray) -> np.ndarray:
    """raw_imu (T, 13_sensors, 13_channels) — уже калиброванные quat+accel+gyro+mag → 305-фичей."""
    per_sensor = []
    for si in range(len(pp.SENSOR_ORDER)):
        r = raw_imu_T_S13[:, si]
        cal_quat, accel, gyro, mag = r[:, :4], r[:, 4:7], r[:, 7:10], r[:, 10:13]
        per_sensor.append(np.concatenate([
            cal_quat, accel, gyro, mag,
            pp.finite_diff(cal_quat), pp.finite_diff(accel), pp.finite_diff(gyro),
        ], axis=1))
    return np.concatenate(per_sensor + [hips, pp.finite_diff(hips)], axis=1).astype(np.float32)


def _model_predict(X: np.ndarray) -> np.ndarray:
    X_norm = (X - ModelBundle.mu) / ModelBundle.sigma
    return ModelBundle.model.predict(X_norm).reshape(-1, 14, 3)


def _build_calibration_dict(text: str) -> dict[str, np.ndarray]:
    """Парсит файл калибровки в словарь {sensor_name: quat_xyzw}."""
    out: dict[str, np.ndarray] = {}
    for line in text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        out[parts[0]] = np.asarray([float(x) for x in parts[1:5]], dtype=np.float32)
    return out


def _parse_sensors_text(text: str) -> tuple[dict[str, np.ndarray], int]:
    """Парсит .sensors-файл из текста (как pp.parse_sensors, но не с диска)."""
    lines = text.splitlines()
    header = lines[0].split()
    num_sensors, num_frames = int(header[0]), int(header[1])
    data: dict[str, list[list[float]]] = {n: [] for n in pp.SENSOR_ORDER}
    idx = 1
    for _ in range(num_frames):
        # пустая строка/номер кадра + num_sensors строк "name v1 v2 ..."
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx < len(lines) and lines[idx].strip().isdigit():
            idx += 1
        for _s in range(num_sensors):
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            parts = lines[idx].split()
            name = parts[0]
            vals = [float(x) for x in parts[1:14]]
            if name in data:
                data[name].append(vals)
            idx += 1
    return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}, num_frames


@app.post("/predict_gif_raw")
def predict_gif_raw(
    sensors: UploadFile = File(..., description=".sensors с показаниями Xsens"),
    calib_bone: UploadFile = File(..., description="<action>_calib_imu_bone.txt"),
    calib_ref: UploadFile = File(..., description="<action>_calib_imu_ref.txt"),
    start: int = Form(0),
    end: int = Form(240),
    fps: int = Form(30),
) -> FileResponse:
    """Принимает сырые файлы TotalCapture (один take), калибрует, гонит через модель, рендерит GIF."""
    if ModelBundle.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if end <= start:
        raise HTTPException(status_code=400, detail="end должен быть больше start")
    try:
        sensors_data, n_total = _parse_sensors_text(sensors.file.read().decode("utf-8", "replace"))
        bone_dict = _build_calibration_dict(calib_bone.file.read().decode("utf-8", "replace"))
        ref_dict = _build_calibration_dict(calib_ref.file.read().decode("utf-8", "replace"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось распарсить файлы: {e}")

    missing = [s for s in pp.SENSOR_ORDER if s not in bone_dict or s not in ref_dict]
    if missing:
        raise HTTPException(status_code=400, detail=f"В калибровках нет сенсоров: {missing}")

    end = min(end, n_total)
    if start >= n_total:
        raise HTTPException(status_code=400, detail=f"start={start} ≥ {n_total} (длина .sensors)")
    n = end - start
    if n > 1200:
        raise HTTPException(status_code=400, detail="Слишком длинный диапазон (>1200 кадров)")

    raw_per_sensor = []
    for s in pp.SENSOR_ORDER:
        sl = sensors_data[s][start:end]
        quat = sl[:, :4]
        accel, gyro, mag = sl[:, 4:7], sl[:, 7:10], sl[:, 10:13]
        cal_quat = pp.calibrate_orientation(quat, ref_dict[s], bone_dict[s])
        raw_per_sensor.append(np.concatenate([cal_quat, accel, gyro, mag], axis=1))
    raw_imu = np.stack(raw_per_sensor, axis=1)            # (n, 13, 13)
    hips = np.zeros((n, 3), dtype=np.float32)             # без BVH восстановить hips нельзя

    X = _features_from_raw_imu(raw_imu, hips)
    pred = _model_predict(X)

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    title = f"raw .sensors frames {start}–{end}  (модель: {ModelBundle.kind})"
    animate(pred, gt=None, out_path=tmp.name, fps=fps, title=title)
    return FileResponse(tmp.name, media_type="image/gif",
                        filename=f"skeleton_raw_{start}_{end}.gif")


@app.post("/predict_gif")
def predict_gif(
    file: UploadFile = File(..., description=".npz из data/processed/ (нужны поля raw_imu, hips; "
                                            "опционально Y для отображения GT)"),
    start: int = Form(0),
    end: int = Form(240),
    fps: int = Form(30),
    show_gt: bool = Form(True),
) -> FileResponse:
    """Принимает .npz, гонит указанный диапазон кадров через модель, рендерит GIF."""
    if ModelBundle.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if end <= start:
        raise HTTPException(status_code=400, detail="end должен быть больше start")

    raw_bytes = file.file.read()
    try:
        data = dict(np.load(io.BytesIO(raw_bytes), allow_pickle=True))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось распарсить .npz: {e}")

    for key in ("raw_imu", "hips"):
        if key not in data:
            raise HTTPException(status_code=400, detail=f"В .npz нет поля '{key}'")

    n_total = data["raw_imu"].shape[0]
    end = min(end, n_total)
    if start >= n_total:
        raise HTTPException(status_code=400, detail=f"start={start} ≥ длины файла {n_total}")
    n = end - start
    if n > 1200:
        raise HTTPException(status_code=400, detail="Слишком длинный диапазон (>1200 кадров)")

    raw = data["raw_imu"][start:end].reshape(n, len(pp.SENSOR_ORDER), 13)
    hips = data["hips"][start:end].astype(np.float32)
    X = _features_from_raw_imu(raw, hips)
    pred = _model_predict(X)

    gt = None
    if show_gt and "Y" in data:
        gt = data["Y"][start:end].reshape(n, 14, 3)

    title = f"frames {start}–{end}"
    if gt is not None:
        mpjpe = float(np.linalg.norm(pred - gt, axis=2).mean())
        title += f"   MPJPE={mpjpe:.2f}   red=pred, green=GT"

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    animate(pred, gt=gt, out_path=tmp.name, fps=fps, title=title)
    return FileResponse(tmp.name, media_type="image/gif",
                        filename=f"skeleton_{start}_{end}.gif")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html lang="ru"><head><meta charset="utf-8"><title>IMU → 3D Skeleton</title>
    <style>
      body { font-family: -apple-system, system-ui, sans-serif; max-width: 820px; margin: 32px auto; padding: 0 16px; color: #222; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 32px; border-bottom: 1px solid #eee; padding-bottom: 4px; }
      code { background: #f3f3f3; padding: 2px 6px; border-radius: 3px; font-size: 0.92em; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-top: 16px; background: #fafafa; }
      label { display: block; margin: 10px 0 4px; font-weight: 500; }
      input[type=number], input[type=file] { width: 100%; padding: 6px 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
      .row { display: flex; gap: 12px; }
      .row > div { flex: 1; }
      button { margin-top: 16px; padding: 10px 18px; background: #0a66c2; color: white; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
      button:hover { background: #084d96; }
      button:disabled { background: #888; cursor: wait; }
      .status { margin-top: 12px; color: #555; font-size: 0.9em; min-height: 1.2em; }
      a { color: #0a66c2; }
      .tabs { display: flex; gap: 2px; margin-top: 16px; }
      .tab { background: #eee; color: #333; padding: 8px 14px; border-radius: 6px 6px 0 0; margin: 0; }
      .tab.active { background: #0a66c2; color: #fff; }
      .tabbox { border-top-left-radius: 0; margin-top: 0; }
    </style></head><body>
    <h1>IMU → 3D Skeleton</h1>
    <p>FastAPI-сервис: 13 IMU-сенсоров → 3D-позиции 14 суставов верхней части тела.
       Модель: <b>MLP</b> (4 residual блока × 512, test MPJPE 1.83 см на TotalCapture S1).</p>

    <div class="tabs">
      <button class="tab active" data-target="tab-npz">Препроцессированный .npz</button>
      <button class="tab" data-target="tab-raw">Сырые .sensors + калибровки</button>
    </div>

    <div id="tab-npz" class="tabbox card">
      <p style="margin-top:0">Загрузите файл из <code>data/processed/</code>
      (нужны поля <code>raw_imu</code>, <code>hips</code>; <code>Y</code> опционально — для GT).</p>
      <form data-endpoint="/predict_gif">
        <label>.npz файл <input type="file" name="file" accept=".npz" required></label>
        <div class="row">
          <div><label>Кадр начала <input type="number" name="start" value="500" min="0"></label></div>
          <div><label>Кадр конца  <input type="number" name="end"   value="740" min="1"></label></div>
          <div><label>FPS         <input type="number" name="fps"   value="30"  min="1" max="60"></label></div>
        </div>
        <label><input type="checkbox" name="show_gt" checked> показать GT-скелет (зелёный)</label>
        <button type="submit">Сгенерировать и скачать GIF</button>
        <div class="status"></div>
      </form>
    </div>

    <div id="tab-raw" class="tabbox card" hidden>
      <p style="margin-top:0">Загрузите три файла одного take из датасета TotalCapture.
        Подробный гайд: <a href="https://github.com/Elur12/hseml-group-project-elur12-1/blob/main/deploy/RAW_DATA_GUIDE.md">RAW_DATA_GUIDE.md</a></p>
      <form data-endpoint="/predict_gif_raw">
        <label><code>&lt;Action&gt;_Xsens_AuxFields.sensors</code><input type="file" name="sensors" accept=".sensors" required></label>
        <label><code>s1_&lt;action&gt;_calib_imu_bone.txt</code><input type="file" name="calib_bone" accept=".txt" required></label>
        <label><code>s1_&lt;action&gt;_calib_imu_ref.txt</code><input type="file" name="calib_ref"  accept=".txt" required></label>
        <div class="row">
          <div><label>Кадр начала <input type="number" name="start" value="500" min="0"></label></div>
          <div><label>Кадр конца  <input type="number" name="end"   value="740" min="1"></label></div>
          <div><label>FPS         <input type="number" name="fps"   value="30"  min="1" max="60"></label></div>
        </div>
        <p style="font-size:0.85em;color:#666">GT (mocap) недоступен без .bvh, поэтому отрисуется только pred.</p>
        <button type="submit">Сгенерировать и скачать GIF</button>
        <div class="status"></div>
      </form>
    </div>

    <h2>Endpoints для интеграции</h2>
    <ul>
      <li><code>POST /predict</code> — JSON с IMU + калибровки → координаты суставов</li>
      <li><code>POST /predict_gif</code> — multipart .npz → GIF</li>
      <li><code>POST /predict_gif_raw</code> — multipart .sensors+calib → GIF</li>
      <li><code>GET /schema</code>, <code>GET /health</code>, <a href="/docs">Swagger UI</a></li>
    </ul>

    <script>
    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.querySelectorAll('.tabbox').forEach(b => b.hidden = b.id !== t.dataset.target);
    }));
    document.querySelectorAll('form[data-endpoint]').forEach(form => {
      form.addEventListener('submit', async e => {
        e.preventDefault();
        const btn = form.querySelector('button');
        const status = form.querySelector('.status');
        btn.disabled = true;
        status.textContent = 'Считаем и рендерим GIF... (~10–30 сек)';
        try {
          const r = await fetch(form.dataset.endpoint, { method: 'POST', body: new FormData(form) });
          if (!r.ok) throw new Error(await r.text());
          const blob = await r.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `skeleton_${form.start.value}_${form.end.value}.gif`;
          document.body.appendChild(a); a.click(); a.remove();
          URL.revokeObjectURL(url);
          status.textContent = '✅ Готово.';
        } catch (err) {
          status.textContent = '❌ ' + err.message;
        } finally { btn.disabled = false; }
      });
    });
    </script>
    </body></html>
    """
