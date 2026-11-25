"""Tahap 1: EDA. Distribusi kelas, file rusak, duplikat, ukuran gambar, dan rasio imbalanced."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src.data import ScanResult, scan_train_dir
from src.utils import fmt_int, info, out_path, save_json, warn_list


def _plot_class_distribution(counts: dict[str, int]) -> Path:
    """Bar chart jumlah gambar per kelas, untuk melihat ketimpangan kelas secara visual."""
    path = out_path("figures", "class_distribution.png")
    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 1.5), 4))
    bars = ax.bar(list(counts), list(counts.values()), color="#6f4e37")
    ax.bar_label(bars)
    ax.set_title("Distribusi Kelas (setelah file rusak & duplikat dibuang)")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah gambar")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_sample_grid(scan: ScanResult) -> Path:
    """Grid contoh gambar acak per kelas (baris = kelas)."""
    path = out_path("figures", "sample_grid.png")
    rng = np.random.default_rng(config.SEED)
    n_cols = config.SAMPLES_PER_CLASS_IN_GRID
    n_rows = len(scan.class_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2), squeeze=False)

    for row, name in enumerate(scan.class_names):
        class_paths = [p for p, y in zip(scan.paths, scan.labels) if y == row]
        chosen = rng.choice(len(class_paths), size=min(n_cols, len(class_paths)), replace=False)
        for col in range(n_cols):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(chosen):
                with Image.open(class_paths[chosen[col]]) as img:
                    ax.imshow(img.convert("RGB"))
        axes[row][0].set_title(name, loc="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def _size_stats(sizes: list[tuple[int, int]]) -> dict[str, dict[str, float]]:
    """Statistik lebar dan tinggi asli; menunjukkan seberapa jauh gambar di-resize."""
    array = np.asarray(sizes)
    return {
        axis: {"min": int(array[:, i].min()), "max": int(array[:, i].max()), "mean": round(float(array[:, i].mean()), 1)}
        for i, axis in enumerate(("width", "height"))
    }


def run_eda(train_dir: Path) -> ScanResult:
    """Jalankan EDA, simpan grafik dan ringkasan JSON, kembalikan data yang sudah bersih."""
    scan = scan_train_dir(train_dir)
    label_counts = Counter(scan.labels)
    counts = {name: label_counts[i] for i, name in enumerate(scan.class_names)}
    # Rasio > ~1.5x menandakan imbalanced; itulah alasan class weight dipakai di tahap 2.
    imbalance_ratio = max(counts.values()) / max(1, min(counts.values()))
    sizes = _size_stats(scan.sizes)

    summary = {
        "class_counts": counts,
        "total_images": len(scan.paths),
        "corrupt_files": [str(p) for p in scan.corrupt],
        "duplicates_removed": [str(p) for p in scan.duplicates],
        "image_size": sizes,
        "imbalance_ratio": round(imbalance_ratio, 2),
    }
    save_json(summary, out_path("reports", "eda_summary.json"))
    dist_path = _plot_class_distribution(counts)
    grid_path = _plot_sample_grid(scan)

    info("Kelas: " + " | ".join(f"{name}={fmt_int(n)}" for name, n in counts.items()))
    info(f"File rusak dilewati: {len(scan.corrupt)} | Duplikat dibuang: {len(scan.duplicates)}")
    warn_list("file rusak dilewati", [str(p) for p in scan.corrupt])
    w, h = sizes["width"], sizes["height"]
    info(f"Ukuran asli: lebar {w['min']}-{w['max']} (rata2 {w['mean']}), tinggi {h['min']}-{h['max']} (rata2 {h['mean']})")
    info(f"Rasio imbalanced: {imbalance_ratio:.1f}x")
    info(f"Grafik: {dist_path}, {grid_path}")
    return scan
