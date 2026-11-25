"""Entry point: menjalankan seluruh pipeline klasifikasi kematangan dalam 6 tahap.

Pemakaian:
    python main.py [--retrain] [--skip-tune] [--test-dir PATH] [--gradcam]
"""

import os

# Diset sebelum TensorFlow/Matplotlib di-import: sembunyikan log info TF dan pakai
# backend Matplotlib tanpa jendela (aplikasi ini murni terminal).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

import config  # noqa: E402
from src.data import PreparedData, ScanResult, prepare_data  # noqa: E402
from src.eda import run_eda  # noqa: E402
from src.evaluate import run_evaluation  # noqa: E402
from src.predict import find_test_samples, run_prediction  # noqa: E402
from src.train import train_all  # noqa: E402
from src.tune import tune_all  # noqa: E402
from src.utils import DataError, info, load_json, out_path, set_seed, stage  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
# Peringatan "retracing" muncul karena tuning sengaja membangun banyak model baru; tidak berbahaya.
tf.get_logger().setLevel("ERROR")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Baca flag CLI."""
    parser = argparse.ArgumentParser(description="Klasifikasi kematangan/kualitas buah atau biji kopi dengan CNN.")
    parser.add_argument("--retrain", action="store_true", help="paksa tuning dan training ulang walau model sudah ada")
    parser.add_argument("--test-dir", metavar="PATH", help=f"folder gambar test (default: {config.TEST_DIR})")
    parser.add_argument("--skip-tune", action="store_true", help="lewati Optuna, pakai DEFAULT_PARAMS di config.py")
    parser.add_argument("--gradcam", action="store_true", help="simpan visualisasi Grad-CAM untuk gambar test")
    return parser.parse_args(argv)


def _check_cached_model(meta: dict, scan: ScanResult) -> None:
    """Pastikan model tersimpan cocok dengan data dan config sekarang."""
    if meta["class_names"] != scan.class_names:
        raise DataError(
            f"Kelas di {config.TRAIN_DIR} {scan.class_names} berbeda dengan model tersimpan {meta['class_names']}.\n"
            "  Petunjuk: jalankan ulang dengan --retrain."
        )
    if tuple(meta["img_size"]) != tuple(config.IMG_SIZE):
        raise DataError(
            f"IMG_SIZE di config.py {config.IMG_SIZE} berbeda dengan model tersimpan {meta['img_size']}.\n"
            "  Petunjuk: kembalikan IMG_SIZE atau jalankan ulang dengan --retrain."
        )


def _warn_if_split_changed(meta: dict, data: PreparedData) -> None:
    """Jika isi data/train berubah, split val berbeda dan bisa berisi gambar yang dulu dilatih."""
    if set(meta.get("val_files", [])) != {str(p) for p in data.val_paths}:
        info("PERINGATAN: data training berubah sejak model dilatih; skor val bisa bocor (terlalu optimis).")
        info("  Jalankan dengan --retrain agar evaluasi val valid.")


def _tune_and_train(args: argparse.Namespace, data: PreparedData) -> dict:
    """Tahap 3 dan 4: cari hyperparameter lalu latih model final."""
    stage(3, "Tuning Optuna")
    if args.skip_tune:
        params = config.DEFAULT_PARAMS
        info("Dilewati (--skip-tune): memakai DEFAULT_PARAMS di config.py")
    else:
        params = tune_all(data)
    stage(4, "Training final")
    return train_all(params, data)


def _display_path(path: Path) -> str:
    """Path relatif terhadap proyek jika bisa, agar cetakan ringkas."""
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def run_pipeline(args: argparse.Namespace, test_dir: Path) -> None:
    """Orkestrasi tahap 1-6."""
    set_seed()
    model_path = out_path("models", "best_model.keras")
    meta_path = out_path("models", "meta.json")
    # Aturan cache: model yang sudah dilatih (misalnya di Colab) langsung dipakai agar demo cepat.
    use_cache = model_path.exists() and meta_path.exists() and not args.retrain

    stage(1, "EDA")
    scan = run_eda(Path(config.TRAIN_DIR))
    cached_meta = load_json(meta_path) if use_cache else None
    if cached_meta:
        _check_cached_model(cached_meta, scan)  # dicek sebelum class_names.json ditimpa tahap 2

    stage(2, "Preprocessing")
    data = prepare_data(scan)

    if cached_meta:
        reason = f"model tersimpan ditemukan ({cached_meta['model_name']}, {cached_meta['trained_at']}); pakai --retrain untuk melatih ulang"
        stage(3, "Tuning Optuna")
        info(f"Dilewati: {reason}")
        stage(4, "Training final")
        info(f"Dilewati: {reason}")
        _warn_if_split_changed(cached_meta, data)
        meta = cached_meta
    else:
        meta = _tune_and_train(args, data)
    model = keras.models.load_model(model_path)

    stage(5, "Evaluasi")
    test = find_test_samples(test_dir, data.class_names, scan.hashes)
    run_evaluation(model, meta, data, test.paths, test.labels)

    stage(6, f"Prediksi ({_display_path(test_dir)})")
    if test.paths:
        run_prediction(model, meta, data.class_names, test, args.gradcam)
    else:
        info("Dilewati: tidak ada gambar test.")
    print("\nSelesai.")


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI; mengembalikan exit code."""
    args = parse_args(argv)
    # --test-dir relatif terhadap folder tempat perintah dijalankan; path lain relatif ke proyek.
    test_dir = Path(args.test_dir).resolve() if args.test_dir else PROJECT_ROOT / config.TEST_DIR
    os.chdir(PROJECT_ROOT)
    try:
        run_pipeline(args, test_dir)
    except DataError as error:
        sys.stdout.flush()  # agar pesan error muncul setelah header tahap, bukan sebelumnya
        print(f"  ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
