[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — IMU → 3D Skeleton Regression

**Студент:** Клюшкин Михаил Александрович

**Группа:** БИВ237


## Оглавление
1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача**: регрессия 3D-позиций суставов верхней части тела (14 суставов) по показаниям 13 IMU-сенсоров (ориентация в кватернионах, акселерометр, гироскоп, магнитометр).

**Датасет**: [TotalCapture](https://cvssp.org/data/totalcapture/) (CVSSP, University of Surrey) — subject S1, 12 записей четырёх типов движений (acting / freestyle / range-of-motion / walking).

**Целевая метрика**: **MPJPE** (Mean Per-Joint Position Error) — среднее евклидово расстояние между предсказанными и истинными 3D-позициями суставов. Вспомогательные: RMSE, MAE.

Подробное описание данных — в [`data/DATA_INFO.md`](data/DATA_INFO.md).


## Структура репозитория
```
.
├── data
│   ├── processed                # Готовые train/val/test.npz и scaler.npz
│   ├── raw                      # Исходные .sensors / .bvh / калибровки (не коммитятся)
│   └── DATA_INFO.md             # Подробное описание данных и сплита
├── models                       # Сохранённые предсказания и метрики моделей
├── deploy
│   ├── Dockerfile               # Образ FastAPI-сервиса
│   ├── docker-compose.yml       # Поднятие API одной командой
│   ├── requirements-api.txt     # Минимум зависимостей для API (без torch)
│   ├── requirements-viz.txt     # Зависимости локального venv для визуализации
│   └── README.md                # Инструкция по запуску деплоя
├── notebooks
│   ├── 01_eda.ipynb             # EDA: размер, распределения, корреляции, скелет, PCA
│   ├── 02_baseline.ipynb        # Baseline: Linear Regression + per-joint MPJPE
│   └── 03_experiments.ipynb     # Сравнение моделей, hyperparam-перебор, ablation
├── presentation                 # Презентация для защиты
├── report
│   ├── images                   # Графики из ноутбуков
│   └── report.md                # Финальный отчёт
├── src
│   ├── preprocessing.py         # Парсеры .sensors/.bvh/калибровок, фичи, сплит
│   ├── modeling.py              # Linear/Ridge/KNN/RF/XGB/LGBM/MLP/BiLSTM + ансамбль
│   ├── skeleton.py              # Топология скелета (suставы и кости)
│   ├── api.py                   # FastAPI-сервис /predict
│   ├── visualize.py             # 3D-визуализация скелета (matplotlib)
│   └── client_example.py        # Демо-клиент API → визуализация
├── tests
│   └── test.py                  # Тесты пайплайна (pytest)
├── requirements.txt
├── pytest.ini
└── README.md
```


## Запуск

```bash
# 1. Виртуальное окружение
python -m venv .venv && source .venv/bin/activate

# 2. Зависимости
pip install -r requirements.txt

# 3. Данные: разложить TotalCapture в data/raw/{s1_imu, s1_Gyro_Mag, S1_vicon}
#    (см. data/DATA_INFO.md)

# 4. Препроцессинг → data/processed/{train,val,test}.npz + scaler.npz
python -m src.preprocessing --data-root data/raw --out data/processed

# 5. Обучение и оценка любой модели
python -m src.modeling --model linear            # baseline
python -m src.modeling --model rf                # RandomForest
python -m src.modeling --model xgb               # XGBoost
python -m src.modeling --model lgbm              # LightGBM
python -m src.modeling --model mlp --epochs 30   # MLP (PyTorch)
python -m src.modeling --model bilstm --epochs 20  # BiLSTM (PyTorch)

# 6. Ансамбль (после обучения нескольких моделей)
python -m src.modeling --model ensemble --members linear rf xgb mlp

# 7. Тесты
pytest

# 8. Линтер
ruff check src/ --line-length 120
```

## Деплой

API (FastAPI + sklearn) запускается в Docker. Визуализатор matplotlib —
отдельный лёгкий локальный `.venv-viz`. Краткая инструкция:
[`deploy/QUICKSTART.md`](deploy/QUICKSTART.md). Подробнее: [`deploy/README.md`](deploy/README.md).

```bash
# 1. API в контейнере
docker compose -f deploy/docker-compose.yml up --build -d
curl http://127.0.0.1:8000/health
# открыть Swagger UI: http://127.0.0.1:8000/docs

# 2. Локальный venv для визуализации
python -m venv .venv-viz && source .venv-viz/bin/activate
pip install -r deploy/requirements-viz.txt

# 3. Демо-клиент: один кадр PNG
python -m src.client_example --start 100 --length 1 --out report/images/deploy_frame.png

# 4. Анимация GIF
python -m src.client_example --start 0 --length 240 --out report/images/deploy_anim.gif
```

## Данные

- `data/raw/` — исходные файлы TotalCapture (см. [`DATA_INFO.md`](data/DATA_INFO.md)). Не коммитятся из-за лицензии.
- `data/processed/` — артефакты препроцессинга:
  - `train.npz` (~31k кадров), `val.npz` (~6k), `test.npz` (~8k)
  - `scaler.npz` — μ/σ стандартизации, имена сенсоров и таргет-суставов

**Сплит** делается по записям (takes), не по кадрам — чтобы избежать data leakage от соседних кадров одной записи. Все 4 категории (acting/freestyle/rom/walking) представлены и в train, и в hold-out.


## Результаты

Метрика MPJPE — в единицах BVH (≈ см для данного скелета). Чем меньше — тем лучше.

| Модель                  | val MPJPE | test MPJPE | test RMSE | test MAE |  time |
|-------------------------|----------:|-----------:|----------:|---------:|------:|
| **Linear (baseline)** ★ |  **0.08** |   **0.11** |      0.11 |     0.04 |    2с |
| KNN (k=8)               |     54.58 |      13.77 |     17.44 |     5.88 |    3с |
| RandomForest            |      5.49 |       1.73 |      2.68 |     0.71 |  6мин |
| XGBoost                 |      3.29 |       1.40 |      1.74 |     0.56 |  4мин |
| LightGBM                |      1.53 |       0.56 |      0.78 |     0.22 |  7мин |
| MLP (4 блока, 20 эп.)   |     11.12 |       5.04 |      4.90 |     2.15 |   24с |
| BiLSTM (12 эп.)         |     57.86 |      14.51 |     13.58 |     6.51 |   60с |
| Ensemble (linear+lgbm)  |      0.77 |       0.31 |       —   |      —   |     — |
| Ensemble (linear+xgb)   |      1.65 |       0.71 |       —   |      —   |     — |
| Ensemble (linear+xgb+rf+lgbm)| 2.19 |      0.80 |       —   |      —   |     — |

> Числа в единицах BVH (≈ см). RMSE и MAE — на test, по плоскому 42-мерному вектору.
>
> **Линейная модель — финальная (★).** MPJPE 0.08–0.11 см невозможно превзойти более сложными моделями: задача почти линейна в пространстве калиброванных кватернионов (позиция сустава = композиция поворотов вышележащих костей с константными длинами). Ансамбли только разбавляют качество Linear. BiLSTM нужен × больше эпох; MLP теряет временной контекст. Подробная интерпретация — в `report/report.md`.

Финальные числа и обоснование выбора модели — в [`report/report.md`](report/report.md).


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md).
