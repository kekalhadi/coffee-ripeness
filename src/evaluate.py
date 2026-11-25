"""Tahap 5: metrik pada val (dan test jika berlabel), confusion matrix, gambar salah tebak."""

import shutil
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config
from src.data import PreparedData, make_base_dataset, make_batches
from src.utils import fmt_int, info, out_path, save_json


def predict_probs(model: keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    """Probabilitas softmax untuk seluruh dataset (urutan sama dengan dataset)."""
    return model.predict(ds, verbose=0)


def predict_paths(model: keras.Model, model_name: str, paths: list[Path]) -> np.ndarray:
    """Probabilitas untuk daftar path gambar, dengan preprocessing yang sama seperti training."""
    # Label dummy (0) karena pipeline butuh pasangan (gambar, label); nilainya tidak dipakai.
    base = make_base_dataset(paths, [0] * len(paths))
    return predict_probs(model, make_batches(base, model_name, config.BATCH_SIZE, training=False))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> tuple[dict, str]:
    """Accuracy, macro-F1, weighted-F1, dan precision/recall/F1 per kelas."""
    # Hanya kelas yang muncul (di label benar atau prediksi) yang dihitung, supaya kelas
    # yang memang tidak ada di folder test tidak dihitung sebagai F1 = 0.
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    names = [class_names[i] for i in present]
    report = classification_report(y_true, y_pred, labels=present, target_names=names, output_dict=True, zero_division=0)
    text = classification_report(y_true, y_pred, labels=present, target_names=names, digits=4, zero_division=0)
    metrics = {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Macro-F1 memberi bobot sama ke tiap kelas sehingga kelas minoritas tetap "terlihat";
        # weighted-F1 memberi bobot sesuai jumlah sampel tiap kelas.
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "per_class": {name: report[name] for name in names},
    }
    return metrics, text


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], split: str) -> Path:
    """Confusion matrix versi jumlah mentah dan ternormalisasi per baris (recall per kelas)."""
    labels = list(range(len(class_names)))
    raw = confusion_matrix(y_true, y_pred, labels=labels)
    # normalize="true": tiap baris dibagi total kelas aslinya, jadi diagonal = recall.
    # Versi ini adil untuk data imbalanced karena kelas besar tidak mendominasi warna.
    normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = max(5, len(class_names) * 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(size * 2, size))
    heatmap_args = {"cmap": "Blues", "xticklabels": class_names, "yticklabels": class_names, "annot": True}
    sns.heatmap(raw, fmt="d", ax=axes[0], **heatmap_args)
    sns.heatmap(normalized, fmt=".2f", vmin=0, vmax=1, ax=axes[1], **heatmap_args)
    for ax, title in zip(axes, ("Jumlah", "Ternormalisasi per baris")):
        ax.set_title(f"{split.upper()} - {title}")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Label benar")
    fig.tight_layout()
    path = out_path("figures", f"confusion_matrix_{split}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_misclassified(paths: list[Path], y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], split: str) -> int:
    """Salin gambar yang salah ditebak (maks MAX_MISCLASSIFIED) untuk analisis kesalahan."""
    wrong = np.flatnonzero(y_true != y_pred)[: config.MAX_MISCLASSIFIED]
    for rank, i in enumerate(wrong):
        name = f"{rank:03d}_true-{class_names[y_true[i]]}_pred-{class_names[y_pred[i]]}_{paths[i].name}"
        shutil.copy2(paths[i], out_path("misclassified", split, name))
    return int((y_true != y_pred).sum())


def evaluate_split(model: keras.Model, model_name: str, paths: list[Path], labels: list[int],
                   class_names: list[str], split: str) -> tuple[dict, str]:
    """Evaluasi satu split (val atau test) dan simpan grafik serta gambar salah tebak."""
    y_true = np.asarray(labels)
    y_pred = predict_paths(model, model_name, paths).argmax(axis=1)
    metrics, text = compute_metrics(y_true, y_pred, class_names)
    plot_confusion(y_true, y_pred, class_names, split)
    n_wrong = save_misclassified(paths, y_true, y_pred, class_names, split)
    return {**metrics, "n_misclassified": n_wrong}, text


def run_evaluation(model: keras.Model, meta: dict, data: PreparedData,
                   test_paths: list[Path], test_labels: list[int] | None) -> dict:
    """Tahap 5: selalu evaluasi val; test hanya jika folder test berlabel."""
    # Hapus hasil run sebelumnya agar folder tidak bercampur gambar dari model lama.
    shutil.rmtree(Path(config.OUTPUT_DIR, "misclassified"), ignore_errors=True)
    # Test set hanya disentuh di sini (evaluasi akhir), tidak pernah di tuning/training.
    splits = [("val", data.val_paths, data.val_labels)]
    if test_labels is not None:
        splits.append(("test", test_paths, test_labels))

    all_metrics, report_parts = {}, []
    for split, paths, labels in splits:
        metrics, text = evaluate_split(model, meta["model_name"], paths, labels, data.class_names, split)
        all_metrics[split] = metrics
        report_parts.append(f"=== {split.upper()} ({fmt_int(len(paths))} gambar, model {meta['model_name']}) ===\n{text}")
        note = " (data test berlabel)" if split == "test" else ""
        info(f"{split.capitalize():<5} -> Accuracy {metrics['accuracy']:.4f} | Macro-F1 {metrics['macro_f1']:.4f} "
             f"| Weighted-F1 {metrics['weighted_f1']:.4f} | Salah: {metrics['n_misclassified']}{note}")
    if test_paths and test_labels is None:
        info("Test  -> dilewati: data test tidak berlabel (hanya prediksi di tahap 6)")

    out_path("reports", "classification_report.txt").write_text("\n".join(report_parts), encoding="utf-8")
    save_json(all_metrics, out_path("reports", "metrics.json"))
    info(f"Laporan: {out_path('reports', 'classification_report.txt')}, {out_path('reports', 'metrics.json')}")
    return all_metrics
