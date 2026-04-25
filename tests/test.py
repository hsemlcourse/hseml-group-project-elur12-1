"""Тесты пайплайна препроцессинга и моделирования.

Запуск:
    pytest tests/ -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src import modeling, preprocessing as pp

DATA_ROOT = "data/raw"
PROCESSED = "data/processed"


def _has_raw_data() -> bool:
    return os.path.exists(os.path.join(DATA_ROOT, "s1_imu", "s1_acting1_Xsens.sensors"))


def _has_processed() -> bool:
    return os.path.exists(os.path.join(PROCESSED, "train.npz"))


needs_raw = pytest.mark.skipif(not _has_raw_data(), reason="raw data not available")
needs_processed = pytest.mark.skipif(not _has_processed(), reason="processed data not available")


# --- Базовые проверки структуры ---------------------------------------------


def test_split_no_overlap():
    train_set = set(pp.SPLIT["train"])
    val_set = set(pp.SPLIT["val"])
    test_set = set(pp.SPLIT["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(pp.ACTIONS)


def test_split_covers_all_categories():
    cats = lambda lst: {a.rstrip("0123456789") for a in lst}
    assert cats(pp.SPLIT["train"]) == {"acting", "freestyle", "rom", "walking"}
    assert cats(pp.SPLIT["val"]) | cats(pp.SPLIT["test"]) == {"acting", "freestyle", "rom", "walking"}


# --- Парсинг калибровки ------------------------------------------------------


@needs_raw
def test_parse_calibration_quat_order():
    paths = pp.take_paths(DATA_ROOT, "acting1")
    calib = pp.parse_calibration(paths.calib_bone)
    assert len(calib) == 13
    for sensor, quat in calib.items():
        assert sensor in pp.SENSOR_ORDER
        assert quat.shape == (4,)
        norm = np.linalg.norm(quat)
        assert 0.95 < norm < 1.05, f"{sensor} quat norm = {norm:.3f}"


def test_xyzw_to_wxyz_conversion():
    q_xyzw = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    q_wxyz = pp.xyzw_to_wxyz(q_xyzw)
    np.testing.assert_array_equal(q_wxyz, np.array([0.9, 0.1, 0.2, 0.3], dtype=np.float32))


# --- Парсинг .sensors --------------------------------------------------------


@needs_raw
def test_parse_sensors_shape():
    paths = pp.take_paths(DATA_ROOT, "acting1")
    sensors, num_frames = pp.parse_sensors(paths.sensors)
    assert num_frames > 0
    assert set(sensors.keys()) == set(pp.SENSOR_ORDER)
    for s, arr in sensors.items():
        assert arr.shape == (num_frames, 13), f"{s} shape={arr.shape}"


# --- Парсинг BVH -------------------------------------------------------------


@needs_raw
def test_parse_bvh_has_target_joints():
    paths = pp.take_paths(DATA_ROOT, "acting1")
    joints, positions, fps = pp.parse_bvh(paths.bvh)
    for j in pp.TARGET_JOINTS + ["Hips"]:
        assert j in joints, f"missing joint {j}"
    assert positions.shape[0] > 0
    assert positions.shape[1] == len(joints)
    assert positions.shape[2] == 3
    assert 50 < fps < 130


# --- Кватернионная алгебра ---------------------------------------------------


def test_quat_identity():
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    res = pp.quat_mul(q, q)
    np.testing.assert_allclose(res, q, atol=1e-6)


def test_quat_conjugate_inverse():
    q = np.array([[0.7071, 0.7071, 0.0, 0.0]], dtype=np.float32)
    qi = pp.quat_conj(q)
    res = pp.quat_mul(q, qi)
    np.testing.assert_allclose(res[0], [1.0, 0.0, 0.0, 0.0], atol=1e-3)


# --- Препроцессинг записи ----------------------------------------------------


@needs_raw
def test_build_take_shapes():
    paths = pp.take_paths(DATA_ROOT, "walking1")
    d = pp.build_take(paths)
    n = d["X"].shape[0]
    assert n > 0
    assert d["Y"].shape == (n, len(pp.TARGET_JOINTS) * 3)
    assert d["hips"].shape == (n, 3)
    # 13 сенсоров × 23 фичи + 6 (hips + hips_vel) = 305
    assert d["X"].shape[1] == 13 * 23 + 6


# --- Processed данные --------------------------------------------------------


@needs_processed
def test_processed_split_consistency():
    train = modeling.load_split("train")
    val = modeling.load_split("val")
    test = modeling.load_split("test")
    assert train["X"].shape[1] == val["X"].shape[1] == test["X"].shape[1]
    assert train["Y"].shape[1] == 14 * 3
    train_acts = set(train["actions"].tolist())
    val_acts = set(val["actions"].tolist())
    test_acts = set(test["actions"].tolist())
    assert train_acts.isdisjoint(val_acts)
    assert train_acts.isdisjoint(test_acts)


@needs_processed
def test_scaler_well_formed():
    s = np.load(os.path.join(PROCESSED, "scaler.npz"), allow_pickle=True)
    mu = s["mu"]
    sigma = s["sigma"]
    assert mu.shape == sigma.shape
    assert (sigma > 0).all()


# --- Метрики -----------------------------------------------------------------


def test_mpjpe_zero_for_identical():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((100, 14 * 3)).astype(np.float32)
    assert modeling.mpjpe(y, y) == pytest.approx(0.0, abs=1e-6)


def test_mpjpe_unit_offset():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((100, 14 * 3)).astype(np.float32)
    y2 = y + 1.0
    val = modeling.mpjpe(y, y2)
    assert val == pytest.approx(np.sqrt(3.0), rel=1e-3)


# --- Воспроизводимость -------------------------------------------------------


@needs_processed
def test_seed_reproducibility_linear():
    m1 = modeling.train_eval("linear", models_dir="/tmp/seed_test_1")
    m2 = modeling.train_eval("linear", models_dir="/tmp/seed_test_2")
    assert m1["val"]["mpjpe"] == pytest.approx(m2["val"]["mpjpe"], rel=1e-9)
    assert m1["test"]["mpjpe"] == pytest.approx(m2["test"]["mpjpe"], rel=1e-9)
