"""Helper umum: seed, path output, JSON, dan format cetakan terminal."""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import config


class DataError(Exception):
    """Kesalahan data/folder yang ditampilkan sebagai pesan singkat, bukan stack trace."""


def set_seed(seed: int = config.SEED) -> None:
    """Kunci seed Python, NumPy, dan TensorFlow agar hasil bisa diulang."""
    # Tanpa seed, split data, inisialisasi bobot, augmentasi, dan sampling Optuna
    # berubah tiap kali dijalankan sehingga hasil tidak bisa dibandingkan.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Seed saja belum cukup: operasi paralel TF (termasuk map paralel di tf.data) bisa
    # berjalan dalam urutan berbeda tiap run. Ini memaksa hasil identik, sedikit lebih lambat.
    tf.config.experimental.enable_op_determinism()


def out_path(*parts: str) -> Path:
    """Path di dalam folder output; folder induknya dibuat otomatis."""
    path = Path(config.OUTPUT_DIR, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder: Path) -> list[Path]:
    """Semua file gambar di folder (rekursif), urut agar deterministik."""
    # File tersembunyi (misalnya "._foto.jpg" buatan macOS) bukan gambar asli.
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTS and not p.name.startswith(".")
    )


def save_json(data: Any, path: Path) -> None:
    """Simpan data ke JSON yang mudah dibaca manusia."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Baca file JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def stage(number: int, title: str) -> None:
    """Cetak header tahap, misalnya "[1/6] EDA"."""
    print(f"\n[{number}/6] {title}")


def info(message: str) -> None:
    """Cetak satu baris ringkasan dengan indentasi."""
    print(f"  {message}")


def warn_list(title: str, items: list[Any]) -> None:
    """Cetak peringatan beserta beberapa contoh item (dibatasi agar terminal tidak banjir)."""
    if not items:
        return
    info(f"PERINGATAN: {title} ({len(items)})")
    for item in items[: config.MAX_PRINTED_WARNINGS]:
        info(f"  - {item}")
    if len(items) > config.MAX_PRINTED_WARNINGS:
        info(f"  ... dan {len(items) - config.MAX_PRINTED_WARNINGS} lainnya")


def fmt_int(number: int) -> str:
    """Format angka gaya Indonesia: 2640 -> "2.640"."""
    return f"{number:,}".replace(",", ".")
