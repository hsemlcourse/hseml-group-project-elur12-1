"""TotalCapture IMU → Skeleton preprocessing.

Парсит исходные файлы датасета TotalCapture и собирает признаки/таргеты для
обучения модели регрессии 3D-позиций суставов верхней части тела.

Источники данных (data/raw):
  - s1_Gyro_Mag/<Action>_Xsens_AuxFields.sensors  — IMU: quat+accel+gyro+mag
  - s1_imu/s1_<action>_calib_imu_{bone,ref}.txt   — калибровки (порядок xyzw)
  - S1_vicon/<action>_BlenderZXY_YmZ.bvh          — ground-truth позиции суставов

Использование:
    python -m src.preprocessing --data-root data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

SEED = 42

SENSOR_ORDER = [
    "Head", "Sternum", "Pelvis",
    "L_UpArm", "R_UpArm", "L_LowArm", "R_LowArm",
    "L_UpLeg", "R_UpLeg", "L_LowLeg", "R_LowLeg",
    "L_Foot", "R_Foot",
]

TARGET_JOINTS = [
    "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
]

ACTIONS = [
    "acting1", "acting2", "acting3",
    "freestyle1", "freestyle2", "freestyle3",
    "rom1", "rom2", "rom3",
    "walking1", "walking2", "walking3",
]

SPLIT = {
    "train": ["acting1", "acting2", "freestyle1", "freestyle2",
              "rom1", "rom2", "walking1", "walking2"],
    "val":   ["acting3", "walking3"],
    "test":  ["freestyle3", "rom3"],
}

FPS = 60.0
DT = 1.0 / FPS


@dataclass
class TakePaths:
    sensors: str
    calib_bone: str
    calib_ref: str
    bvh: str


def take_paths(data_root: str, action: str) -> TakePaths:
    return TakePaths(
        sensors=os.path.join(data_root, "s1_Gyro_Mag", f"{action.capitalize()}_Xsens_AuxFields.sensors"),
        calib_bone=os.path.join(data_root, "s1_imu", f"s1_{action}_calib_imu_bone.txt"),
        calib_ref=os.path.join(data_root, "s1_imu", f"s1_{action}_calib_imu_ref.txt"),
        bvh=os.path.join(data_root, "S1_vicon", f"{action}_BlenderZXY_YmZ.bvh"),
    )


def parse_sensors(path: str) -> tuple[dict[str, np.ndarray], int]:
    """Парсит .sensors c полями: quat(wxyz, 4) + accel(3) + gyro(3) + mag(3) = 13."""
    with open(path, "r") as f:
        lines = f.readlines()
    header = lines[0].split()
    num_sensors, num_frames = int(header[0]), int(header[1])
    data: dict[str, list[list[float]]] = {n: [] for n in SENSOR_ORDER}
    idx = 1
    for _ in range(num_frames):
        idx += 1  # пропустить строку с номером кадра
        for _ in range(num_sensors):
            parts = lines[idx].strip().split()
            name = parts[0]
            vals = [float(x) for x in parts[1:]]
            if name in data:
                data[name].append(vals)
            idx += 1
    out = {n: np.asarray(v, dtype=np.float32) for n, v in data.items()}
    return out, num_frames


def parse_calibration(path: str) -> dict[str, np.ndarray]:
    """Парсит файл калибровки. Кватернион в формате (x y z w)."""
    with open(path, "r") as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    calib: dict[str, np.ndarray] = {}
    for i in range(1, n + 1):
        parts = lines[i].strip().split()
        calib[parts[0]] = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
    return calib


def parse_bvh(path: str) -> tuple[list[str], np.ndarray, float]:
    """Возвращает (joint_names, positions(N, J, 3), fps).

    В этом BVH каждый сустав имеет 6 каналов: Xposition Yposition Zposition
    Zrotation Xrotation Yrotation — поэтому глобальные позиции берём напрямую
    из первых трёх каналов каждого сустава, без forward kinematics.
    """
    with open(path, "r") as f:
        content = f.read()
    hier_part, motion_part = content.split("MOTION")

    joint_names: list[str] = []
    channels_per_joint: list[int] = []
    for line in hier_part.split("\n"):
        s = line.strip()
        if s.startswith("ROOT") or s.startswith("JOINT"):
            joint_names.append(s.split()[-1])
        elif s.startswith("CHANNELS"):
            channels_per_joint.append(int(s.split()[1]))
    # CHANNELS у End Site нет, поэтому длина списка совпадает с числом суставов
    assert len(channels_per_joint) == len(joint_names), \
        f"channels={len(channels_per_joint)} joints={len(joint_names)}"

    motion_lines = motion_part.strip().split("\n")
    frame_time = 1.0 / FPS
    data_start = 0
    for i, line in enumerate(motion_lines):
        if line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1].strip())
            data_start = i + 1
            break

    frames: list[np.ndarray] = []
    for line in motion_lines[data_start:]:
        s = line.strip()
        if not s:
            continue
        vals = [float(x) for x in s.split()]
        positions = np.zeros((len(joint_names), 3), dtype=np.float32)
        offset = 0
        for j, nch in enumerate(channels_per_joint):
            positions[j] = vals[offset:offset + 3]
            offset += nch
        frames.append(positions)
    fps = 1.0 / frame_time if frame_time > 0 else FPS
    return joint_names, np.asarray(frames, dtype=np.float32), fps


# --- Quaternion utilities (порядок wxyz) -------------------------------------


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def calibrate_orientation(imu_quat_wxyz: np.ndarray,
                          calib_ref_xyzw: np.ndarray,
                          calib_bone_xyzw: np.ndarray) -> np.ndarray:
    """R^g_b = R_ref^-1 · R_imu · R_bone (все в формате wxyz)."""
    q_ref = xyzw_to_wxyz(calib_ref_xyzw)
    q_bone = xyzw_to_wxyz(calib_bone_xyzw)
    q_ref_inv = quat_conj(q_ref)
    tmp = quat_mul(q_ref_inv[None, :], imu_quat_wxyz)
    return quat_mul(tmp, q_bone[None, :])


def finite_diff(signal: np.ndarray, dt: float = DT) -> np.ndarray:
    vel = np.zeros_like(signal)
    if signal.shape[0] < 2:
        return vel
    vel[1:] = (signal[1:] - signal[:-1]) / dt
    vel[0] = vel[1]
    return vel


# --- Feature/target построение ----------------------------------------------


def build_take(paths: TakePaths) -> dict[str, np.ndarray]:
    """Парсит одну запись и возвращает словарь массивов фичей/таргетов."""
    sensors, n_imu_frames = parse_sensors(paths.sensors)
    calib_bone = parse_calibration(paths.calib_bone)
    calib_ref = parse_calibration(paths.calib_ref)
    joint_names, gt_pos, _ = parse_bvh(paths.bvh)

    n = min(n_imu_frames, gt_pos.shape[0])

    per_sensor_feats: list[np.ndarray] = []
    raw_per_sensor: list[np.ndarray] = []
    for s in SENSOR_ORDER:
        raw = sensors[s][:n]
        quat = raw[:, :4]
        accel = raw[:, 4:7]
        gyro = raw[:, 7:10]
        mag = raw[:, 10:13]
        cal_quat = calibrate_orientation(quat, calib_ref[s], calib_bone[s])
        feats = np.concatenate([
            cal_quat, accel, gyro, mag,
            finite_diff(cal_quat), finite_diff(accel), finite_diff(gyro),
        ], axis=1)
        per_sensor_feats.append(feats)
        raw_per_sensor.append(np.concatenate([cal_quat, accel, gyro, mag], axis=1))

    hips_idx = joint_names.index("Hips")
    hips = gt_pos[:n, hips_idx]
    X = np.concatenate(per_sensor_feats + [hips, finite_diff(hips)], axis=1)

    target_idx = [joint_names.index(j) for j in TARGET_JOINTS]
    Y = gt_pos[:n][:, target_idx, :] - hips[:, None, :]
    Y = Y.reshape(n, -1)

    raw_imu = np.concatenate(raw_per_sensor, axis=1)  # для EDA: 13×13

    return {
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "hips": hips.astype(np.float32),
        "raw_imu": raw_imu.astype(np.float32),
    }


def build_split(data_root: str, actions: list[str]) -> dict[str, np.ndarray]:
    Xs, Ys, hips, raws, take_ids, frame_ids = [], [], [], [], [], []
    for take_id, action in enumerate(actions):
        paths = take_paths(data_root, action)
        d = build_take(paths)
        n = d["X"].shape[0]
        Xs.append(d["X"])
        Ys.append(d["Y"])
        hips.append(d["hips"])
        raws.append(d["raw_imu"])
        take_ids.append(np.full(n, take_id, dtype=np.int32))
        frame_ids.append(np.arange(n, dtype=np.int32))
        print(f"  {action}: {n} frames, X={d['X'].shape[1]}, Y={d['Y'].shape[1]}")
    return {
        "X": np.concatenate(Xs, axis=0),
        "Y": np.concatenate(Ys, axis=0),
        "hips": np.concatenate(hips, axis=0),
        "raw_imu": np.concatenate(raws, axis=0),
        "take_id": np.concatenate(take_ids, axis=0),
        "frame_id": np.concatenate(frame_ids, axis=0),
        "actions": np.asarray(actions),
    }


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0).astype(np.float32)
    sigma = X.std(axis=0).astype(np.float32)
    sigma[sigma < 1e-6] = 1.0
    return mu, sigma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(SEED)

    print("[train]")
    train = build_split(args.data_root, SPLIT["train"])
    print("[val]")
    val = build_split(args.data_root, SPLIT["val"])
    print("[test]")
    test = build_split(args.data_root, SPLIT["test"])

    mu, sigma = fit_standardizer(train["X"])
    for split_name, d in [("train", train), ("val", val), ("test", test)]:
        d["X_norm"] = ((d["X"] - mu) / sigma).astype(np.float32)
        np.savez_compressed(
            os.path.join(args.out, f"{split_name}.npz"),
            X=d["X"], X_norm=d["X_norm"], Y=d["Y"],
            hips=d["hips"], raw_imu=d["raw_imu"],
            take_id=d["take_id"], frame_id=d["frame_id"], actions=d["actions"],
        )
        print(f"  saved {split_name}: {d['X'].shape[0]} frames")

    np.savez_compressed(
        os.path.join(args.out, "scaler.npz"),
        mu=mu, sigma=sigma,
        target_joints=np.asarray(TARGET_JOINTS),
        sensor_order=np.asarray(SENSOR_ORDER),
    )
    print(f"  saved scaler: mu/sigma shape={mu.shape}")


if __name__ == "__main__":
    main()
