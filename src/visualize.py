"""3D-визуализация скелета верхней части тела через matplotlib.

Используется из CLI (картинка/анимация) и из клиента API.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.skeleton import EDGES, JOINTS


def _setup_ax(ax, joints_3d: np.ndarray) -> None:
    pad = 0.2 * (joints_3d.max() - joints_3d.min() + 1e-6)
    lo = joints_3d.min(axis=(0, 1)) - pad
    hi = joints_3d.max(axis=(0, 1)) + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))


def plot_skeleton(joints: np.ndarray, ax=None, color="steelblue",
                  joint_labels: bool = False, title: str | None = None):
    """joints: (14, 3). Возвращает (fig, ax)."""
    assert joints.shape == (14, 3), f"expected (14,3), got {joints.shape}"
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    _setup_ax(ax, joints[None, :, :])
    for i, j in EDGES:
        xs = [joints[i, 0], joints[j, 0]]
        ys = [joints[i, 1], joints[j, 1]]
        zs = [joints[i, 2], joints[j, 2]]
        ax.plot(xs, ys, zs, color=color, lw=2.5)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=30)
    if joint_labels:
        for k, name in enumerate(JOINTS):
            ax.text(joints[k, 0], joints[k, 1], joints[k, 2], name, fontsize=7)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_compare(pred: np.ndarray, gt: np.ndarray | None = None, title: str | None = None):
    """Сравнение pred и (опционально) ground truth в одной системе координат."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    stacked = pred[None, :, :] if gt is None else np.stack([pred, gt])
    _setup_ax(ax, stacked)
    if gt is not None:
        for i, j in EDGES:
            ax.plot([gt[i, 0], gt[j, 0]], [gt[i, 1], gt[j, 1]], [gt[i, 2], gt[j, 2]],
                    color="green", lw=2, alpha=0.6, label="GT" if (i, j) == EDGES[0] else None)
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c="green", s=20, alpha=0.6)
    for i, j in EDGES:
        ax.plot([pred[i, 0], pred[j, 0]], [pred[i, 1], pred[j, 1]], [pred[i, 2], pred[j, 2]],
                color="crimson", lw=2.5, label="Pred" if (i, j) == EDGES[0] else None)
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="crimson", s=30)
    if gt is not None:
        ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    return fig, ax


def animate(frames: np.ndarray, gt: np.ndarray | None = None,
            out_path: str | None = None, fps: int = 30, title: str | None = None):
    """frames: (T, 14, 3). Если gt задан той же формы — рисует оба скелета (pred=red, GT=green).
    Сохраняет GIF, если задан out_path."""
    assert frames.ndim == 3 and frames.shape[1:] == (14, 3), f"got {frames.shape}"
    if gt is not None:
        assert gt.shape == frames.shape, f"gt {gt.shape} != pred {frames.shape}"
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    bounds = frames if gt is None else np.concatenate([frames, gt], axis=0)
    _setup_ax(ax, bounds)
    if title:
        ax.set_title(title)

    pred_color = "crimson" if gt is not None else "steelblue"
    pred_lines = [ax.plot([], [], [], color=pred_color, lw=2.5,
                          label="Pred" if (gt is not None and k == 0) else None)[0]
                  for k in range(len(EDGES))]
    pred_scat = ax.scatter([], [], [], c=pred_color, s=30)
    gt_lines: list = []
    gt_scat = None
    if gt is not None:
        gt_lines = [ax.plot([], [], [], color="green", lw=2, alpha=0.6,
                            label="GT" if k == 0 else None)[0]
                    for k in range(len(EDGES))]
        gt_scat = ax.scatter([], [], [], c="green", s=20, alpha=0.6)
        ax.legend(loc="upper left")

    def _draw(skel, lines, scat):
        for (i, j), line in zip(EDGES, lines):
            line.set_data([skel[i, 0], skel[j, 0]], [skel[i, 1], skel[j, 1]])
            line.set_3d_properties([skel[i, 2], skel[j, 2]])
        scat._offsets3d = (skel[:, 0], skel[:, 1], skel[:, 2])

    def update(t: int):
        _draw(frames[t], pred_lines, pred_scat)
        artists = [*pred_lines, pred_scat]
        if gt is not None:
            _draw(gt[t], gt_lines, gt_scat)
            artists += [*gt_lines, gt_scat]
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    if out_path:
        anim.save(out_path, writer="pillow", fps=fps)
    return anim


def main() -> None:
    parser = argparse.ArgumentParser(description="Визуализация предсказаний скелета")
    parser.add_argument("--input", required=True, help=".npy/.npz с массивом (T,14,3) или (14,3)")
    parser.add_argument("--out", default=None, help="Файл для сохранения (png/gif)")
    parser.add_argument("--gt", default=None, help="Опционально: .npy с ground truth той же формы")
    parser.add_argument("--frame", type=int, default=None, help="Один кадр вместо анимации")
    args = parser.parse_args()

    data = np.load(args.input)
    if hasattr(data, "files"):
        data = data[data.files[0]]
    if data.ndim == 1:
        data = data.reshape(-1, 14, 3)
    if data.ndim == 2:
        data = data.reshape(1, 14, 3)

    if args.frame is not None or len(data) == 1:
        frame = data[args.frame or 0]
        gt_frame = np.load(args.gt).reshape(-1, 14, 3)[args.frame or 0] if args.gt else None
        plot_compare(frame, gt_frame, title="Predicted skeleton")
        if args.out:
            plt.savefig(args.out, dpi=120, bbox_inches="tight")
            print(f"saved → {args.out}")
        else:
            plt.show()
    else:
        animate(data, out_path=args.out, title="Predicted skeleton")
        if args.out:
            print(f"saved → {args.out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
