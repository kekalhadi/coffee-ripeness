"""Tes end-to-end: python main.py pada dataset sintetis kecil (sekitar 30 detik)."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import config
import main

COLORS = {"overripe": (70, 35, 20), "ripe": (200, 30, 30), "unripe": (60, 160, 60)}


def _make_images(folder: Path, color: tuple[int, int, int], count: int, rng: np.random.Generator) -> None:
    folder.mkdir(parents=True)
    for i in range(count):
        img = Image.fromarray(rng.integers(180, 255, (64, 64, 3), dtype=np.uint8))
        x, y = rng.integers(20, 44, size=2)
        ImageDraw.Draw(img).ellipse([x - 15, y - 15, x + 15, y + 15], fill=color)
        img.save(folder / f"{i:02d}.png")


@pytest.fixture
def small_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rng = np.random.default_rng(0)
    for name, color in COLORS.items():
        _make_images(tmp_path / "train" / name, color, 12, rng)
        _make_images(tmp_path / "test" / name, color, 2, rng)
    monkeypatch.setattr(config, "TRAIN_DIR", str(tmp_path / "train"))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(config, "IMG_SIZE", (96, 96))
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "OPTUNA_TRIALS", 2)
    monkeypatch.setattr(config, "TUNE_EPOCHS", 1)
    return tmp_path


def test_full_pipeline_then_cached_run(small_project: Path) -> None:
    outputs = small_project / "outputs"
    test_dir = str(small_project / "test")

    assert main.main(["--test-dir", test_dir]) == 0
    first_meta = json.loads((outputs / "models" / "meta.json").read_text())
    for expected in ("predictions.csv", "reports/metrics.json", "reports/best_params.json",
                     "reports/model_comparison.csv", "figures/confusion_matrix_test.png"):
        assert (outputs / expected).exists(), expected

    # Run kedua tanpa --retrain harus memakai model tersimpan (meta.json tidak berubah).
    assert main.main(["--test-dir", test_dir]) == 0
    assert json.loads((outputs / "models" / "meta.json").read_text()) == first_meta
    metrics = json.loads((outputs / "reports" / "metrics.json").read_text())
    assert {"val", "test"} <= metrics.keys()
