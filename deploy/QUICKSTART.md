# Quickstart — деплой за 3 шага

## 1. Поднять API
```bash
docker compose -f deploy/docker-compose.yml up --build -d
curl http://127.0.0.1:8000/health
```
Swagger UI: <http://127.0.0.1:8000/docs>

## 2. Подготовить визуализатор (один раз)
```bash
python -m venv .venv-viz
source .venv-viz/bin/activate
pip install -r deploy/requirements-viz.txt
```

## 3. Получить скелет
```bash
# одна поза → PNG
python -m src.client_example --start 100 --length 1 --out frame.png

# движение → GIF
python -m src.client_example --start 0 --length 240 --out anim.gif
```

## Остановить
```bash
docker compose -f deploy/docker-compose.yml down
```

## Свои данные

POST `http://127.0.0.1:8000/predict` JSON:
```json
{
  "frames": [
    {
      "Head":     {"quat":[w,x,y,z], "accel":[ax,ay,az], "gyro":[gx,gy,gz], "mag":[mx,my,mz]},
      "Sternum":  {"quat":[...], "accel":[...], "gyro":[...], "mag":[...]}
      // ... остальные 11 сенсоров (см. GET /schema)
    }
  ],
  "calibration": {
    "Head":    {"ref":[x,y,z,w], "bone":[x,y,z,w]},
    "Sternum": {"ref":[...],     "bone":[...]}
    // ... по одной калибровке на сенсор
  },
  "hips": [[x,y,z]]    // опционально
}
```

Ответ:
```json
{
  "joints": ["Spine", "Spine1", ..., "LeftHand"],
  "edges":  [[0,1],[1,2],...],
  "positions": [[[x,y,z], ...14 joints]],
  "n_frames": 1
}
```
