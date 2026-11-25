"""Tes cepat untuk validasi data, duplikat, split, dan deteksi folder test."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import compute_class_weights, scan_train_dir, split_train_val
from src.predict import find_test_samples
from src.utils import DataError


def _save_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


@pytest.fixture
def train_dir(tmp_path: Path) -> Path:
    root = tmp_path / "train"
    for label, name in enumerate(("ripe", "unripe")):
        for i in range(10):
            _save_image(root / name / f"{i}.png", label * 100 + i)
    (root / "ripe" / "broken.jpg").write_bytes(b"bukan gambar")
    shutil.copy(root / "ripe" / "0.png", root / "ripe" / "copy.png")
    return root


def test_scan_skips_corrupt_and_duplicate_files(train_dir: Path) -> None:
    scan = scan_train_dir(train_dir)

    assert scan.class_names == ["ripe", "unripe"]
    assert len(scan.paths) == 20
    assert [p.name for p in scan.corrupt] == ["broken.jpg"]
    assert [p.name for p in scan.duplicates] == ["copy.png"]


def test_scan_missing_folder_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="tidak ditemukan"):
        scan_train_dir(tmp_path / "tidak_ada")


def test_split_is_stratified_and_disjoint(train_dir: Path) -> None:
    scan = scan_train_dir(train_dir)

    train_p, val_p, _, val_y = split_train_val(scan.paths, scan.labels)

    assert not set(train_p) & set(val_p)
    assert sorted(set(val_y)) == [0, 1]


def test_class_weights_favor_minority_class() -> None:
    weights = compute_class_weights([0, 0, 0, 1], n_classes=2)

    assert weights[1] > weights[0]


def test_test_dir_labeled_when_subfolders_match_classes(tmp_path: Path) -> None:
    _save_image(tmp_path / "ripe" / "a.png", 1)
    _save_image(tmp_path / "unripe" / "b.png", 2)

    test = find_test_samples(tmp_path, ["ripe", "unripe"], frozenset())

    assert test.labels == [0, 1]


def test_test_dir_unlabeled_when_images_are_loose(tmp_path: Path) -> None:
    _save_image(tmp_path / "a.png", 1)

    test = find_test_samples(tmp_path, ["ripe", "unripe"], frozenset())

    assert len(test.paths) == 1 and test.labels is None


def test_missing_or_empty_test_dir_returns_empty(tmp_path: Path) -> None:
    assert find_test_samples(tmp_path / "tidak_ada", ["ripe"], frozenset()).paths == []
    assert find_test_samples(tmp_path, ["ripe"], frozenset()).paths == []
