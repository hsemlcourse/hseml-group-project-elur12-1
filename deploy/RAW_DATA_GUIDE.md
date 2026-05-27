# Гайд: загрузка сырых данных с сенсоров

Сервис умеет принимать **сырые файлы датасета TotalCapture** (без предварительного
препроцессинга в `.npz`). Используйте вкладку **«Сырые .sensors + калибровки»**
на главной странице или `POST /predict_gif_raw`.

## Какие файлы нужны

Для одного take (например, `acting1`) — **три файла**:

| Файл | Откуда взять | Что внутри |
|------|--------------|-----------|
| `Acting1_Xsens_AuxFields.sensors` | `s1_Gyro_Mag/` | Показания 13 IMU за все кадры записи: для каждого сенсора `quat[wxyz]` + accel + gyro + mag |
| `s1_acting1_calib_imu_bone.txt` | `s1_imu/` | Калибровочный кватернион `[x y z w]` для каждого сенсора — поворот кости относительно IMU |
| `s1_acting1_calib_imu_ref.txt`  | `s1_imu/` | Калибровочный кватернион `[x y z w]` — ориентация IMU в reference-позе |

`.bvh` (mocap ground truth) **не нужен** — сервис делает только предсказание, без
сравнения с GT.

## Где взять данные

[TotalCapture](https://cvssp.org/data/totalcapture/) (CVSSP, University of Surrey) —
открытый датасет под research-лицензией. После регистрации скачиваются архивы по
субъектам; для нашего проекта — subject S1.

После распаковки структура:
```
data/raw/
├── s1_imu/                # калибровки + .sensors (только quat+accel)
│   ├── s1_acting1_Xsens.sensors
│   ├── s1_acting1_calib_imu_bone.txt
│   └── s1_acting1_calib_imu_ref.txt
├── s1_Gyro_Mag/           # .sensors с полным набором (quat+accel+gyro+mag)
│   └── Acting1_Xsens_AuxFields.sensors
└── S1_vicon/              # BVH-файлы mocap (опционально)
    └── acting1_BlenderZXY_YmZ.bvh
```

Сервису нужен **именно `_Xsens_AuxFields.sensors`** из `s1_Gyro_Mag/` — он содержит
полный набор каналов (gyro+mag, которые в `s1_imu/`-версии отсутствуют).

## Формат `.sensors`

```
13 7398                       ← num_sensors num_frames
1                              ← номер кадра
Head w x y z ax ay az gx gy gz mx my mz
Sternum w x y z ax ay az gx gy gz mx my mz
...
Pelvis w x y z ax ay az gx gy gz mx my mz
2
Head w x y z ...
...
```

- `num_sensors=13`, `num_frames` — общее число кадров (~2000–7000 на take).
- Кватернионы IMU в порядке **`(w, x, y, z)`** (scalar-first).
- Сенсоры идут в фиксированном порядке: Head, Sternum, Pelvis, L_UpArm, R_UpArm,
  L_LowArm, R_LowArm, L_UpLeg, R_UpLeg, L_LowLeg, R_LowLeg, L_Foot, R_Foot.

## Формат калибровок

Каждая строка — `<sensor_name> x y z w`:

```
Head     0.123 -0.456 0.789 0.321
Sternum  0.987  0.654 0.321 0.012
...
```

Кватернионы здесь в порядке **`(x, y, z, w)`** (scalar-last) — это типичная путаница,
сервис учитывает её при калибровке.

## Что делает сервис при загрузке

1. Парсит `.sensors` → массив `(num_frames, 13_sensors, 13_channels)`.
2. По каждой строке калибровок строит словарь `{sensor → quat}`.
3. Для каждого сенсора применяет калибровку:
   `R^global = R_ref⁻¹ · R_imu · R_bone` (см. `src.preprocessing.calibrate_orientation`).
4. Собирает 305 фичей (как при офлайн-препроцессинге): калиброванный кватернион +
   accel + gyro + mag + velocity-фичи. Hips заполняется нулями (без BVH восстановить
   глобальную позицию таза нельзя).
5. Стандартизация по сохранённому `scaler.npz`.
6. Инференс MLP → `(T, 14, 3)` позиций суставов в hip-relative координатах.
7. Рендер GIF через matplotlib.

## Пример с curl

```bash
curl -X POST http://127.0.0.1:8000/predict_gif_raw \
  -F "sensors=@data/raw/s1_Gyro_Mag/Acting1_Xsens_AuxFields.sensors" \
  -F "calib_bone=@data/raw/s1_imu/s1_acting1_calib_imu_bone.txt" \
  -F "calib_ref=@data/raw/s1_imu/s1_acting1_calib_imu_ref.txt" \
  -F "start=500" -F "end=740" -F "fps=30" \
  -o skeleton.gif
```

## Ограничения

- Только subject S1: модель обучена на нём. На других субъектах TotalCapture
  предсказание возможно, но качество не гарантируется (нет кросс-субъектной CV).
- Максимум 1200 кадров за один запрос — иначе рендер GIF становится медленным
  (>30 сек) и съедает много памяти контейнера.
- Без `.bvh` нет ground-truth, поэтому в анимации только pred (красный). Чтобы
  сравнивать с GT — используйте вкладку «.npz» с препроцессированным файлом.