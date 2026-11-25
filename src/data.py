"""Validasi gambar, deteksi duplikat, split stratified, pipeline tf.data, augmentasi, class weight."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from src.utils import DataError, fmt_int, info, list_images, out_path, save_json


@dataclass(frozen=True)
class ScanResult:
    """Hasil pemindaian data/train setelah file rusak dan duplikat dibuang."""

    class_names: list[str]
    paths: list[Path]
    labels: list[int]
    sizes: list[tuple[int, int]]  # (lebar, tinggi) asli tiap gambar valid
    corrupt: list[Path]
    duplicates: list[Path]
    hashes: frozenset[str]  # MD5 gambar training, untuk cek kebocoran ke folder test


@dataclass(frozen=True)
class PreparedData:
    """Data train/val yang siap dipakai tuning, training, dan evaluasi."""

    class_names: list[str]
    train_paths: list[Path]
    train_labels: list[int]
    val_paths: list[Path]
    val_labels: list[int]
    class_weight: dict[int, float]
    train_base: tf.data.Dataset  # belum di-batch, sudah di-cache
    val_base: tf.data.Dataset


# ---------------------------------------------------------------- validasi & duplikat

def read_image_size(path: Path) -> tuple[int, int] | None:
    """Decode gambar sampai selesai; kembalikan (lebar, tinggi) atau None jika rusak."""
    # verify() saja tidak cukup: file yang terpotong bisa lolos verify tetapi gagal
    # saat di-decode di tengah training. Jadi gambar benar-benar di-load sekali di sini.
    try:
        with Image.open(path) as img:
            img.convert("RGB").load()
            return img.size
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None


def file_md5(path: Path) -> str:
    """Hash MD5 isi file; dua file dengan hash sama adalah duplikat persis."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _find_class_names(train_dir: Path) -> list[str]:
    """Nama kelas = nama subfolder yang berisi gambar, urut abjad (tidak ditulis manual)."""
    if not train_dir.is_dir():
        raise DataError(
            f"Folder data training tidak ditemukan: {train_dir}\n"
            f"  Petunjuk: buat {train_dir}/<nama_kelas>/ lalu isi tiap folder dengan gambar."
        )
    class_names = sorted(
        d.name for d in train_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and list_images(d)
    )
    if len(class_names) < 2:
        raise DataError(
            f"Butuh minimal 2 subfolder kelas berisi gambar di {train_dir}, ditemukan {len(class_names)}.\n"
            f"  Contoh: {train_dir}/ripe/ dan {train_dir}/unripe/"
        )
    return class_names


def scan_train_dir(train_dir: Path) -> ScanResult:
    """Pindai data training: buang file rusak dan duplikat persis (MD5)."""
    class_names = _find_class_names(train_dir)
    paths, labels, sizes, corrupt, duplicates = [], [], [], [], []
    seen_hashes: set[str] = set()

    for label, name in enumerate(class_names):
        for path in list_images(train_dir / name):
            size = read_image_size(path)
            if size is None:
                corrupt.append(path)
                continue
            digest = file_md5(path)
            # Duplikat dibuang SEBELUM split. Kalau tidak, salinan yang sama bisa masuk
            # train dan val sekaligus sehingga skor val terlihat bagus padahal "nyontek".
            if digest in seen_hashes:
                duplicates.append(path)
                continue
            seen_hashes.add(digest)
            paths.append(path)
            labels.append(label)
            sizes.append(size)

    return ScanResult(class_names, paths, labels, sizes, corrupt, duplicates, frozenset(seen_hashes))


# ---------------------------------------------------------------- split & class weight

def split_train_val(paths: list[Path], labels: list[int]) -> tuple[list, list, list, list]:
    """Split stratified train/val dari daftar path (sebelum augmentasi apa pun)."""
    # Stratified: proporsi tiap kelas di val sama dengan di train, penting untuk data imbalanced.
    try:
        train_p, val_p, train_y, val_y = train_test_split(
            paths, labels, test_size=config.VAL_SPLIT, stratify=labels, random_state=config.SEED
        )
    except ValueError as error:
        raise DataError(
            f"Split train/val gagal: {error}\n"
            "  Petunjuk: tiap kelas butuh minimal 2 gambar, dan jumlah gambar val "
            "harus >= jumlah kelas. Tambah gambar atau perbesar VAL_SPLIT di config.py."
        ) from error
    return list(train_p), list(val_p), list(train_y), list(val_y)


def compute_class_weights(labels: list[int], n_classes: int) -> dict[int, float]:
    """Bobot kelas 'balanced': kelas yang sedikit diberi bobot loss lebih besar."""
    # Tanpa class weight, model cenderung menebak kelas mayoritas saja karena itu
    # sudah menurunkan loss. Bobot = n_sampel / (n_kelas * n_sampel_kelas).
    classes = np.arange(n_classes)
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------- pipeline tf.data

def load_resized_image(path: Path | str) -> np.ndarray:
    """Baca gambar sebagai RGB uint8 berukuran IMG_SIZE (dipakai train DAN prediksi)."""
    # Satu fungsi yang sama untuk training dan prediksi menjamin preprocessing identik.
    # Pillow dipakai (bukan tf.io.decode_image) karena TensorFlow tidak bisa membaca .webp.
    height, width = config.IMG_SIZE
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((width, height)), dtype=np.uint8)


def get_preprocess_fn(model_name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Normalisasi piksel sesuai kebutuhan model."""
    if model_name == "mobilenetv2":
        # MobileNetV2 dilatih di ImageNet dengan piksel [-1, 1]; bobot pretrained hanya
        # bermakna jika input kita diskalakan dengan cara yang sama.
        return keras.applications.mobilenet_v2.preprocess_input
    # Baseline dilatih dari nol: cukup skala 1/255 ke [0, 1] agar gradien stabil.
    # Skala tetap ini tidak memakai statistik val/test, jadi tidak ada kebocoran.
    return lambda images: images / 255.0


def build_augmenter() -> keras.Sequential:
    """Augmentasi ringan yang meniru variasi foto nyata (posisi, jarak, cahaya)."""
    # Variasi sengaja kecil: rotasi/zoom besar bisa memotong objek, dan perubahan warna
    # tidak dipakai karena warna justru ciri utama tingkat kematangan.
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=config.SEED),
            keras.layers.RandomRotation(config.AUG_ROTATION, seed=config.SEED),
            keras.layers.RandomZoom(config.AUG_ZOOM, seed=config.SEED),
            keras.layers.RandomBrightness(config.AUG_BRIGHTNESS, value_range=(0, 255), seed=config.SEED),
            keras.layers.RandomContrast(config.AUG_CONTRAST, value_range=(0, 255), seed=config.SEED),
        ],
        name="augmenter",
    )


def make_base_dataset(paths: list[Path], labels: list[int]) -> tf.data.Dataset:
    """Dataset (gambar uint8, label) yang belum di-batch, di-cache di memori."""
    ds = tf.data.Dataset.from_tensor_slices(([str(p) for p in paths], np.asarray(labels, np.int32)))

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(lambda p: load_resized_image(p.decode()), [path], tf.uint8)
        image.set_shape((*config.IMG_SIZE, 3))
        return image, label

    # cache() setelah resize: decode JPEG cukup sekali, epoch berikutnya baca dari memori.
    # Disimpan sebagai uint8 (4x lebih hemat daripada float32).
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()


def make_batches(base: tf.data.Dataset, model_name: str, batch_size: int, training: bool) -> tf.data.Dataset:
    """Batch + (augmentasi jika training) + normalisasi + prefetch."""
    ds = base
    if training:
        # Shuffle hanya untuk train: urutan val/test harus tetap agar cocok dengan daftar path.
        ds = ds.shuffle(int(base.cardinality()), seed=config.SEED)
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y))
    if training:
        # Augmentasi HANYA di train: val/test harus mencerminkan gambar asli.
        augmenter = build_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    preprocess = get_preprocess_fn(model_name)
    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # prefetch: CPU menyiapkan batch berikutnya selagi model memproses batch sekarang.
    return ds.prefetch(tf.data.AUTOTUNE)


def prepare_data(scan: ScanResult) -> PreparedData:
    """Tahap 2: split, class weight, dan dataset dasar untuk train/val."""
    train_p, val_p, train_y, val_y = split_train_val(scan.paths, scan.labels)
    class_weight = compute_class_weights(train_y, len(scan.class_names))
    save_json(scan.class_names, out_path("models", "class_names.json"))

    weights_text = ", ".join(f"{scan.class_names[c]}={w:.2f}" for c, w in class_weight.items())
    info(f"Train: {fmt_int(len(train_p))} | Val: {fmt_int(len(val_p))} | Ukuran: {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}")
    info(f"Class weight: {weights_text}")

    return PreparedData(
        class_names=scan.class_names,
        train_paths=train_p,
        train_labels=train_y,
        val_paths=val_p,
        val_labels=val_y,
        class_weight=class_weight,
        train_base=make_base_dataset(train_p, train_y),
        val_base=make_base_dataset(val_p, val_y),
    )
