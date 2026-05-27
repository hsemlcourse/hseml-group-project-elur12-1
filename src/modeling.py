"""Модели и обучение для регрессии IMU → 3D-позиции суставов.

Модели:
  - linear   : Linear Regression (baseline, без feature engineering сверх стандартизации)
  - knn      : KNeighborsRegressor
  - rf       : RandomForestRegressor
  - xgb      : XGBoost (multi-output через MultiOutputRegressor)
  - lgbm     : LightGBM (multi-output через MultiOutputRegressor)
  - mlp      : MLP на одиночных кадрах (PyTorch)
  - bilstm   : BiLSTM на последовательностях (PyTorch)
  - ensemble : Усреднение предсказаний выбранных моделей

Метрика: MPJPE (мм) — основная; RMSE/MAE — вспомогательные.

Использование:
    python -m src.modeling --model linear
    python -m src.modeling --model bilstm --epochs 30
    python -m src.modeling --model ensemble --members linear rf bilstm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np

SEED = 42
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
NUM_TARGET_JOINTS = 14


# --- Воспроизводимость -------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --- Метрики -----------------------------------------------------------------


def mpjpe(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Per-Joint Position Error (мм). y shape: (N, J*3)."""
    n = y_pred.shape[0]
    p = y_pred.reshape(n, NUM_TARGET_JOINTS, 3)
    t = y_true.reshape(n, NUM_TARGET_JOINTS, 3)
    return float(np.linalg.norm(p - t, axis=2).mean())


def per_joint_mpjpe(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = y_pred.shape[0]
    p = y_pred.reshape(n, NUM_TARGET_JOINTS, 3)
    t = y_true.reshape(n, NUM_TARGET_JOINTS, 3)
    return np.linalg.norm(p - t, axis=2).mean(axis=0)


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


@dataclass
class Metrics:
    mpjpe: float
    rmse: float
    mae: float

    @classmethod
    def compute(cls, y_pred: np.ndarray, y_true: np.ndarray) -> "Metrics":
        return cls(mpjpe(y_pred, y_true), rmse(y_pred, y_true), mae(y_pred, y_true))


# --- Загрузка данных ---------------------------------------------------------


def load_split(split: str, processed_dir: str = PROCESSED_DIR) -> dict[str, np.ndarray]:
    return dict(np.load(os.path.join(processed_dir, f"{split}.npz"), allow_pickle=True))


# --- Sklearn-совместимые модели ---------------------------------------------


def make_model(name: str, **kwargs):
    name = name.lower()
    if name == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1)
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=kwargs.get("alpha", 1.0), random_state=SEED)
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=kwargs.get("n_neighbors", 8),
                                   weights="distance", n_jobs=-1)
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 20),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            n_jobs=-1, random_state=SEED,
        )
    if name == "xgb":
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        base = XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0,
        )
        return MultiOutputRegressor(base, n_jobs=1)
    if name == "lgbm":
        from sklearn.multioutput import MultiOutputRegressor
        from lightgbm import LGBMRegressor
        base = LGBMRegressor(
            n_estimators=kwargs.get("n_estimators", 300),
            num_leaves=kwargs.get("num_leaves", 63),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=SEED, n_jobs=-1, verbosity=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)
    raise ValueError(f"Unknown sklearn model: {name}")


# --- PyTorch модели ----------------------------------------------------------


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLPRegressor:
    """MLP на одиночных кадрах. Sklearn-подобный интерфейс fit/predict."""

    def __init__(self, hidden=512, num_blocks=4, dropout=0.15,
                 epochs=30, batch_size=512, lr=1e-3, device=None):
        self.hidden = hidden
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
        self.history = []

    def _build(self, in_dim: int, out_dim: int):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, in_dim, out_dim, hidden, num_blocks, dropout):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                    nn.GELU(), nn.Dropout(dropout),
                )
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                        nn.GELU(), nn.Dropout(dropout),
                        nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
                    ) for _ in range(num_blocks)
                ])
                self.head = nn.Sequential(
                    nn.GELU(), nn.Linear(hidden, hidden // 2),
                    nn.GELU(), nn.Linear(hidden // 2, out_dim),
                )

            def forward(self, x):
                h = self.proj(x)
                for b in self.blocks:
                    h = torch.relu(h + b(h))
                return self.head(h)

        return _Net(in_dim, out_dim, self.hidden, self.num_blocks, self.dropout)

    def fit(self, X, Y, X_val=None, Y_val=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        set_seed(SEED)
        device = self.device or get_device()
        self.model = self._build(X.shape[1], Y.shape[1]).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = torch.nn.SmoothL1Loss()

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float()),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
        )
        for ep in range(self.epochs):
            self.model.train()
            tot = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * xb.size(0)
            sched.step()
            avg = tot / len(loader.dataset)
            val_mpjpe = None
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_mpjpe = mpjpe(val_pred, Y_val)
            self.history.append({"epoch": ep + 1, "train_loss": avg, "val_mpjpe": val_mpjpe})
            print(f"  [mlp] ep {ep+1:02d} loss={avg:.4f}"
                  + (f" val_mpjpe={val_mpjpe:.2f}" if val_mpjpe is not None else ""))
        return self

    def predict(self, X):
        import torch
        device = self.device or get_device()
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).float().to(device)
                outs.append(self.model(xb).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def save(self, path: str, in_dim: int, out_dim: int) -> None:
        """Сохраняет архитектуру + веса в один .pt-файл."""
        import torch
        torch.save({
            "state_dict": self.model.state_dict(),
            "config": {
                "in_dim": in_dim, "out_dim": out_dim,
                "hidden": self.hidden, "num_blocks": self.num_blocks, "dropout": self.dropout,
                "batch_size": self.batch_size,
            },
        }, path)

    @classmethod
    def load(cls, path: str, device=None) -> "MLPRegressor":
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        obj = cls(hidden=cfg["hidden"], num_blocks=cfg["num_blocks"],
                  dropout=cfg["dropout"], batch_size=cfg.get("batch_size", 512),
                  device=device)
        obj.model = obj._build(cfg["in_dim"], cfg["out_dim"])
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.to(device or get_device())
        obj.model.eval()
        return obj


class BiLSTMRegressor:
    """BiLSTM на последовательностях кадров. fit/predict работают пакадрово,
    обучение использует overlapping окна, предсказание — sliding window с
    усреднением перекрытий."""

    def __init__(self, hidden=192, num_layers=2, dropout=0.3,
                 seq_len=64, stride=16, epochs=20, batch_size=64,
                 lr=1e-3, device=None):
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.stride = stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.model = None
        self.history = []

    def _build(self, in_dim: int, out_dim: int):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, in_dim, out_dim, hidden, num_layers, dropout):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.LayerNorm(hidden),
                    nn.GELU(), nn.Dropout(dropout),
                )
                self.lstm = nn.LSTM(
                    hidden, hidden, num_layers=num_layers, batch_first=True,
                    bidirectional=True, dropout=dropout if num_layers > 1 else 0,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2 + hidden, hidden), nn.LayerNorm(hidden),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, out_dim),
                )

            def forward(self, x):
                h0 = self.proj(x)
                h, _ = self.lstm(h0)
                h = torch.cat([h, h0], dim=-1)
                return self.head(h)

        return _Net(in_dim, out_dim, self.hidden, self.num_layers, self.dropout)

    def _windows(self, take_id: np.ndarray, stride: int):
        """Окна внутри одной записи (без склейки между takes)."""
        starts = []
        for tid in np.unique(take_id):
            mask = take_id == tid
            idx = np.where(mask)[0]
            n = len(idx)
            for s in range(0, n - self.seq_len + 1, stride):
                starts.append((idx[0] + s, idx[0] + s + self.seq_len))
        return starts

    def fit(self, X, Y, take_id, X_val=None, Y_val=None, take_id_val=None):
        import torch
        set_seed(SEED)
        device = self.device or get_device()
        self.model = self._build(X.shape[1], Y.shape[1]).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = torch.nn.SmoothL1Loss()

        starts = self._windows(take_id, self.stride)
        X_t = torch.from_numpy(X).float()
        Y_t = torch.from_numpy(Y).float()

        for ep in range(self.epochs):
            self.model.train()
            random.shuffle(starts)
            tot = 0.0
            n_batches = 0
            for i in range(0, len(starts), self.batch_size):
                batch = starts[i:i + self.batch_size]
                xb = torch.stack([X_t[s:e] for s, e in batch]).to(device)
                yb = torch.stack([Y_t[s:e] for s, e in batch]).to(device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                n_batches += 1
            sched.step()
            avg = tot / max(n_batches, 1)
            val_mpjpe = None
            if X_val is not None:
                val_pred = self.predict(X_val, take_id_val)
                val_mpjpe = mpjpe(val_pred, Y_val)
            self.history.append({"epoch": ep + 1, "train_loss": avg, "val_mpjpe": val_mpjpe})
            print(f"  [bilstm] ep {ep+1:02d} loss={avg:.4f}"
                  + (f" val_mpjpe={val_mpjpe:.2f}" if val_mpjpe is not None else ""))
        return self

    def predict(self, X, take_id):
        import torch
        device = self.device or get_device()
        self.model.eval()
        out = np.zeros((X.shape[0], self.model.head[-1].out_features), dtype=np.float32)
        cnt = np.zeros(X.shape[0], dtype=np.float32)
        with torch.no_grad():
            for tid in np.unique(take_id):
                idx = np.where(take_id == tid)[0]
                n = len(idx)
                stride = max(1, self.seq_len // 4)
                if n < self.seq_len:
                    # padding: повторим хвост
                    pad = self.seq_len - n
                    Xw = np.concatenate([X[idx], np.repeat(X[idx[-1:]], pad, axis=0)])
                    pred = self.model(torch.from_numpy(Xw).float().unsqueeze(0).to(device))
                    out[idx] = pred[0, :n].cpu().numpy()
                    cnt[idx] = 1.0
                    continue
                for s in range(0, n - self.seq_len + 1, stride):
                    xb = torch.from_numpy(X[idx[s:s + self.seq_len]]).float().unsqueeze(0).to(device)
                    pred = self.model(xb).cpu().numpy()[0]
                    out[idx[s:s + self.seq_len]] += pred
                    cnt[idx[s:s + self.seq_len]] += 1.0
                # хвост
                if (n - self.seq_len) % stride != 0:
                    s = n - self.seq_len
                    xb = torch.from_numpy(X[idx[s:s + self.seq_len]]).float().unsqueeze(0).to(device)
                    pred = self.model(xb).cpu().numpy()[0]
                    out[idx[s:s + self.seq_len]] += pred
                    cnt[idx[s:s + self.seq_len]] += 1.0
        out /= np.maximum(cnt[:, None], 1.0)
        return out


# --- Тренировка/оценка на датасете -----------------------------------------


def train_eval(model_name: str, processed_dir: str = PROCESSED_DIR,
               models_dir: str = MODELS_DIR, **kwargs) -> dict:
    set_seed(SEED)
    os.makedirs(models_dir, exist_ok=True)
    train = load_split("train", processed_dir)
    val = load_split("val", processed_dir)
    test = load_split("test", processed_dir)

    Xtr, Ytr = train["X_norm"], train["Y"]
    Xv, Yv = val["X_norm"], val["Y"]
    Xte, Yte = test["X_norm"], test["Y"]

    t0 = time.time()
    if model_name in ("linear", "ridge", "knn", "rf", "xgb", "lgbm"):
        model = make_model(model_name, **kwargs)
        model.fit(Xtr, Ytr)
        pred_val = model.predict(Xv)
        pred_test = model.predict(Xte)
    elif model_name == "mlp":
        model = MLPRegressor(**kwargs)
        model.fit(Xtr, Ytr, X_val=Xv, Y_val=Yv)
        pred_val = model.predict(Xv)
        pred_test = model.predict(Xte)
        model.save(os.path.join(models_dir, "mlp.pt"), in_dim=Xtr.shape[1], out_dim=Ytr.shape[1])
    elif model_name == "bilstm":
        model = BiLSTMRegressor(**kwargs)
        model.fit(Xtr, Ytr, train["take_id"],
                  X_val=Xv, Y_val=Yv, take_id_val=val["take_id"])
        pred_val = model.predict(Xv, val["take_id"])
        pred_test = model.predict(Xte, test["take_id"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    train_time = time.time() - t0

    metrics = {
        "model": model_name,
        "train_time_sec": round(train_time, 1),
        "val": asdict(Metrics.compute(pred_val, Yv)),
        "test": asdict(Metrics.compute(pred_test, Yte)),
        "kwargs": kwargs,
    }
    print(f"[{model_name}] val MPJPE={metrics['val']['mpjpe']:.2f}  "
          f"test MPJPE={metrics['test']['mpjpe']:.2f}  ({train_time:.1f}s)")

    np.savez_compressed(
        os.path.join(models_dir, f"pred_{model_name}.npz"),
        val=pred_val, test=pred_test,
    )
    with open(os.path.join(models_dir, f"metrics_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def ensemble(members: list[str], processed_dir: str = PROCESSED_DIR,
             models_dir: str = MODELS_DIR) -> dict:
    val_preds, test_preds = [], []
    for m in members:
        p = np.load(os.path.join(models_dir, f"pred_{m}.npz"))
        val_preds.append(p["val"])
        test_preds.append(p["test"])
    val_pred = np.mean(val_preds, axis=0)
    test_pred = np.mean(test_preds, axis=0)
    val = load_split("val", processed_dir)
    test = load_split("test", processed_dir)
    metrics = {
        "model": "ensemble:" + "+".join(members),
        "val": asdict(Metrics.compute(val_pred, val["Y"])),
        "test": asdict(Metrics.compute(test_pred, test["Y"])),
    }
    print(f"[ensemble] val MPJPE={metrics['val']['mpjpe']:.2f}  "
          f"test MPJPE={metrics['test']['mpjpe']:.2f}")
    with open(os.path.join(models_dir, "metrics_ensemble.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["linear", "ridge", "knn", "rf", "xgb", "lgbm",
                                 "mlp", "bilstm", "ensemble"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--members", nargs="+", default=None)
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    args = parser.parse_args()

    if args.model == "ensemble":
        if not args.members:
            raise SystemExit("--members required for ensemble")
        ensemble(args.members, args.processed_dir, args.models_dir)
        return

    kwargs: dict = {}
    if args.epochs is not None and args.model in ("mlp", "bilstm"):
        kwargs["epochs"] = args.epochs
    train_eval(args.model, args.processed_dir, args.models_dir, **kwargs)


if __name__ == "__main__":
    main()
