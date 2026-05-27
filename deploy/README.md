# Deploy

API (FastAPI + sklearn) живёт в Docker. Визуализатор matplotlib —
отдельный лёгкий локальный `.venv-viz`, чтобы не тащить heavyweight зависимости
в host для рисования картинок и общения с сервисом.

## 1. Запуск API в Docker

Из корня репозитория:

```bash
docker compose -f deploy/docker-compose.yml up --build -d
# проверка
curl http://127.0.0.1:8000/health
open http://127.0.0.1:8000/docs   # Swagger UI
```

Внутри образа лежит обученная `models/linear.joblib` (если её нет — модель
дообучается на старте за ~1 сек из `data/processed/train.npz`).

Остановить:

```bash
docker compose -f deploy/docker-compose.yml down
```

## 2. Локальный venv для визуализации

```bash
python -m venv .venv-viz
source .venv-viz/bin/activate
pip install -r deploy/requirements-viz.txt
```

Один кадр PNG:
```bash
python -m src.client_example --start 100 --length 1 --out report/images/deploy_frame.png
```

Анимация GIF:
```bash
python -m src.client_example --start 0 --length 240 --out report/images/deploy_anim.gif
```

## 3. Формат входных данных

См. `GET /schema` или `POST /predict` в Swagger UI. Краткая структура:

```jsonc
{
  "frames": [
    {
      "Head":     {"quat":[w,x,y,z], "accel":[..3..], "gyro":[..3..], "mag":[..3..]},
      "Sternum":  {...},
      // ... ещё 11 сенсоров
    }
    // ещё кадры
  ],
  "calibration": {
    "Head":    {"ref":[x,y,z,w], "bone":[x,y,z,w]},
    "Sternum": {...}
  },
  "hips": [[x,y,z], ...]   // опционально
}
```

Ответ:

```jsonc
{
  "joints":   ["Spine", "Spine1", ..., "LeftHand"],
  "edges":    [[0,1],[1,2],...],
  "positions":[[[x,y,z], ...14 joints], ...N frames],
  "n_frames": N
}
```
