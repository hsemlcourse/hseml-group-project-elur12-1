"""Демо-клиент: берёт окно кадров из test.npz, отправляет в API, визуализирует ответ.

Запуск (требует запущенный API на 127.0.0.1:8000):
    uvicorn src.api:app --port 8000 &
    python -m src.client_example --start 0 --length 1   # один кадр → PNG
    python -m src.client_example --start 0 --length 120 --out demo.gif
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np

from src import preprocessing as pp
from src.visualize import animate, plot_compare

IDENTITY_XYZW = [0.0, 0.0, 0.0, 1.0]  # калибровка-плейсхолдер (raw_imu уже калиброван)


def build_request(processed_dir: str, start: int, length: int) -> tuple[dict, np.ndarray]:
    """Из test.npz формируем JSON по схеме API + возвращаем ground-truth для визуализации."""
    test = dict(np.load(f"{processed_dir}/test.npz", allow_pickle=True))
    raw = test["raw_imu"][start:start + length]   # (T, 13*13)
    hips = test["hips"][start:start + length].tolist()
    Y_gt = test["Y"][start:start + length].reshape(-1, 14, 3)

    raw = raw.reshape(-1, len(pp.SENSOR_ORDER), 13)
    frames = []
    for t in range(raw.shape[0]):
        frame = {}
        for si, sname in enumerate(pp.SENSOR_ORDER):
            row = raw[t, si]
            frame[sname] = {
                "quat": row[:4].tolist(),    # уже калиброванный wxyz
                "accel": row[4:7].tolist(),
                "gyro": row[7:10].tolist(),
                "mag": row[10:13].tolist(),
            }
        frames.append(frame)
    calibration = {s: {"ref": IDENTITY_XYZW, "bone": IDENTITY_XYZW} for s in pp.SENSOR_ORDER}
    payload = {"frames": frames, "calibration": calibration, "hips": hips}
    return payload, Y_gt


def call_api(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=1, help="число кадров (1 → png; >1 → анимация)")
    parser.add_argument("--out", default=None, help="png / gif. Если не задан — plt.show().")
    args = parser.parse_args()

    print(f"→ собираем {args.length} кадров из test.npz (start={args.start})")
    payload, Y_gt = build_request(args.processed_dir, args.start, args.length)
    print(f"→ POST {args.url}  payload≈{len(json.dumps(payload))//1024} KiB")
    resp = call_api(args.url, payload)
    pred = np.asarray(resp["positions"], dtype=np.float32)
    print(f"← получено {resp['n_frames']} кадров, ошибка MPJPE на этом окне: "
          f"{float(np.linalg.norm(pred - Y_gt, axis=2).mean()):.3f}")

    if args.length == 1:
        plot_compare(pred[0], Y_gt[0], title=f"frame {args.start}: red=pred, green=GT")
        if args.out:
            import matplotlib.pyplot as plt
            plt.savefig(args.out, dpi=120, bbox_inches="tight")
            print(f"saved → {args.out}")
        else:
            import matplotlib.pyplot as plt
            plt.show()
    else:
        animate(pred, out_path=args.out, title=f"frames {args.start}–{args.start+args.length}")
        if args.out:
            print(f"saved → {args.out}")
        else:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    main()
