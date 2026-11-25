"""Tahap 6: deteksi folder test (berlabel/tidak), prediksi batch, CSV, dan Grad-CAM opsional."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from src.data import file_md5, get_preprocess_fn, load_resized_image, read_image_size
from src.evaluate import predict_paths
from src.utils import fmt_int, info, list_images, out_path, warn_list


@dataclass(frozen=True)
class TestSet:
    """Gambar test yang valid; labels=None berarti folder test tidak berlabel."""

    root: Path
    paths: list[Path]
    labels: list[int] | None


def _collect_test_pairs(test_dir: Path, class_names: list[str]) -> tuple[list[tuple[Path, int]], bool]:
    """Kumpulkan (path, label). Berlabel jika ada subfolder yang namanya sama dengan nama kelas."""
    subdirs = {d.name for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")}
    labeled_classes = [name for name in class_names if name in subdirs]
    if not labeled_classes:
        return [(p, -1) for p in list_images(test_dir)], False
    warn_list("subfolder test bukan nama kelas, diabaikan", sorted(subdirs - set(class_names)))
    pairs = [(p, class_names.index(name)) for name in labeled_classes for p in list_images(test_dir / name)]
    return pairs, True


def find_test_samples(test_dir: Path, class_names: list[str], train_hashes: frozenset[str]) -> TestSet:
    """Deteksi struktur folder test dan buang file rusak. Tidak pernah crash jika folder kosong."""
    empty = TestSet(test_dir, [], None)
    if not test_dir.is_dir():
        info(f"PERINGATAN: folder test tidak ditemukan: {test_dir} (evaluasi test & prediksi dilewati)")
        return empty

    pairs, is_labeled = _collect_test_pairs(test_dir, class_names)
    corrupt = {p for p, _ in pairs if read_image_size(p) is None}
    warn_list("file test rusak dilewati", sorted(str(p) for p in corrupt))
    valid = [(p, y) for p, y in pairs if p not in corrupt]
    if not valid:
        info(f"PERINGATAN: tidak ada gambar valid di {test_dir} (evaluasi test & prediksi dilewati)")
        return empty

    # Gambar test yang identik dengan gambar training membuat skor test terlalu optimis.
    leaked = [str(p) for p, _ in valid if file_md5(p) in train_hashes]
    warn_list("gambar test identik dengan gambar training (skor test jadi terlalu optimis)", leaked)
    kind = "berlabel" if is_labeled else "tidak berlabel"
    info(f"Folder test: {fmt_int(len(valid))} gambar, {kind}")
    return TestSet(test_dir, [p for p, _ in valid], [y for _, y in valid] if is_labeled else None)


def _print_predictions(df: pd.DataFrame) -> None:
    """Cetak maksimal MAX_PRINTED_PREDICTIONS baris lalu ringkasan jumlah per kelas."""
    shown = df.head(config.MAX_PRINTED_PREDICTIONS)
    width = min(40, int(shown["filename"].str.len().max()))
    for row in shown.itertuples():
        info(f"{row.filename:<{width}} -> {row.predicted_class:<12} ({row.confidence:.1%})")
    if len(df) > len(shown):
        info(f"... {fmt_int(len(df) - len(shown))} gambar lainnya ada di CSV")
    counts = df["predicted_class"].value_counts()
    info("Ringkasan: " + " | ".join(f"{name}={fmt_int(n)}" for name, n in counts.items()))


def compute_gradcam(grad_model: keras.Model, image: tf.Tensor, class_index: int) -> np.ndarray:
    """Heatmap Grad-CAM bernilai 0..1 seukuran peta fitur konvolusi terakhir."""
    # Ide Grad-CAM: gradien skor kelas terhadap peta fitur menunjukkan seberapa penting tiap
    # channel. Peta fitur dijumlahkan dengan bobot itu, sehingga area "panas" adalah bagian
    # gambar yang paling mendorong model memilih kelas tersebut.
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(image, training=False)
        class_score = predictions[:, class_index]
    grads = tape.gradient(class_score, feature_maps)
    channel_weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(feature_maps[0] * channel_weights, axis=-1))
    return (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()


def save_gradcams(model: keras.Model, meta: dict, test: TestSet, predicted: np.ndarray, class_names: list[str]) -> Path:
    """Simpan overlay Grad-CAM untuk tiap gambar test."""
    target_dir = Path(config.OUTPUT_DIR, "figures", "gradcam")
    shutil.rmtree(target_dir, ignore_errors=True)  # buang heatmap dari run sebelumnya
    target_dir.mkdir(parents=True)
    layer = model.get_layer(meta["gradcam_layer"])
    grad_model = keras.Model(model.inputs, [layer.output, model.output])
    preprocess = get_preprocess_fn(meta["model_name"])

    for i, path in enumerate(test.paths):
        raw = load_resized_image(path)
        image = preprocess(tf.convert_to_tensor(raw[None], dtype=tf.float32))
        heatmap = compute_gradcam(grad_model, image, int(predicted[i]))
        heatmap = tf.image.resize(heatmap[..., None], config.IMG_SIZE).numpy()[..., 0]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(raw)
        ax.imshow(heatmap, cmap="jet", alpha=0.4)
        ax.set_title(f"{path.name}\nprediksi: {class_names[predicted[i]]}", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(target_dir / f"{i:04d}_{path.stem}.png", dpi=100)
        plt.close(fig)
    return target_dir


def run_prediction(model: keras.Model, meta: dict, class_names: list[str], test: TestSet, gradcam: bool) -> pd.DataFrame:
    """Tahap 6: prediksi batch, cetak ringkasan, simpan predictions.csv (dan Grad-CAM)."""
    probs = predict_paths(model, meta["model_name"], test.paths)
    predicted = probs.argmax(axis=1)
    columns = {
        "filename": [str(p.relative_to(test.root)) for p in test.paths],
        "predicted_class": [class_names[i] for i in predicted],
        "confidence": probs.max(axis=1).round(4),
        **{f"prob_{name}": probs[:, i].round(4) for i, name in enumerate(class_names)},
    }
    if test.labels is not None:
        columns = {
            **columns,
            "true_class": [class_names[i] for i in test.labels],
            "correct": predicted == np.asarray(test.labels),
        }
    df = pd.DataFrame(columns)
    csv_path = out_path("predictions.csv")
    df.to_csv(csv_path, index=False)

    _print_predictions(df)
    info(f"Hasil lengkap: {csv_path}")
    if gradcam:
        info(f"Grad-CAM: {save_gradcams(model, meta, test, predicted, class_names)}")
    return df
