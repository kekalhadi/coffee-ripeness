"""Tahap 4: training final kedua model, kurva loss/accuracy, dan pemilihan model terbaik."""

import json
import shutil
from datetime import datetime

import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config
from src.data import PreparedData, make_batches
from src.evaluate import predict_probs
from src.models import GRADCAM_LAYERS, build_model
from src.utils import info, out_path, save_json


def early_stopping(patience: int) -> keras.callbacks.EarlyStopping:
    """Hentikan training saat val_loss berhenti membaik."""
    # Setelah titik itu model mulai menghafal data train (overfitting). restore_best_weights
    # mengembalikan bobot dari epoch terbaik, bukan dari epoch terakhir.
    return keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)


def fit_model(model_name: str, params: dict, data: PreparedData, epochs: int,
              callbacks: list[keras.callbacks.Callback], verbose: int = 0) -> tuple[keras.Model, dict]:
    """Bangun dan latih satu model; dipakai bersama oleh tuning dan training final."""
    model = build_model(model_name, len(data.class_names), params)
    batch_size = params["batch_size"]
    history = model.fit(
        make_batches(data.train_base, model_name, batch_size, training=True),
        validation_data=make_batches(data.val_base, model_name, batch_size, training=False),
        epochs=epochs,
        shuffle=False,  # sudah di-shuffle di make_batches (hanya train)
        # class_weight memperbesar loss untuk kelas minoritas (lihat data.compute_class_weights).
        class_weight=data.class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history.history


def score_on_val(model: keras.Model, model_name: str, data: PreparedData, batch_size: int) -> dict[str, float]:
    """Accuracy dan macro-F1 pada val."""
    # Macro-F1 dipakai untuk memilih model karena, berbeda dengan accuracy, nilainya tetap
    # jelek jika model mengabaikan kelas minoritas.
    probs = predict_probs(model, make_batches(data.val_base, model_name, batch_size, training=False))
    y_pred = probs.argmax(axis=1)
    return {
        "val_accuracy": float(accuracy_score(data.val_labels, y_pred)),
        "val_macro_f1": float(f1_score(data.val_labels, y_pred, average="macro", zero_division=0)),
    }


def _plot_curves(history: dict, model_name: str) -> None:
    """Kurva train vs val; jarak yang melebar antara keduanya = tanda overfitting."""
    epochs = range(1, len(history["loss"]) + 1)
    for metric, file_prefix in (("loss", "loss_curve"), ("accuracy", "accuracy_curve")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, history[metric], marker="o", label="train")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label="val")
        ax.set_title(f"{metric.capitalize()} - {model_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path("figures", f"{file_prefix}_{model_name}.png"), dpi=120)
        plt.close(fig)


def train_final(model_name: str, params: dict, data: PreparedData) -> dict:
    """Latih satu model dengan hyperparameter terbaik dan nilai hasilnya pada val."""
    checkpoint = out_path("models", f"{model_name}.keras")
    callbacks = [
        early_stopping(config.EARLY_STOP_PATIENCE),
        # Simpan ke disk hanya saat val_loss membaik, jadi file selalu berisi versi terbaik.
        keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
        # Saat val_loss mandek, langkah belajar diperkecil agar bisa turun lebih halus ke minimum.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE, min_lr=config.MIN_LEARNING_RATE,
        ),
        keras.callbacks.CSVLogger(str(out_path("reports", f"train_log_{model_name}.csv"))),
    ]
    info(f"Melatih {model_name} (maks {config.EPOCHS} epoch) ...")
    _, history = fit_model(model_name, params, data, config.EPOCHS, callbacks, verbose=2)
    _plot_curves(history, model_name)

    # Nilai model dari file checkpoint, yaitu file yang benar-benar akan dipakai prediksi.
    best_model = keras.models.load_model(checkpoint)
    scores = score_on_val(best_model, model_name, data, params["batch_size"])
    return {
        "model": model_name,
        **scores,
        "epochs_run": len(history["loss"]),
        "best_val_loss": float(min(history["val_loss"])),
        "total_params": best_model.count_params(),
        "trainable_params": sum(int(w.numpy().size) for w in best_model.trainable_weights),
        "params": params,
        "checkpoint": str(checkpoint),
    }


def train_all(params_by_model: dict[str, dict], data: PreparedData) -> dict:
    """Latih semua model, bandingkan macro-F1 val, simpan yang terbaik dan meta.json."""
    results = []
    for name in config.MODEL_NAMES:
        keras.backend.clear_session()  # buang graf model sebelumnya dari memori
        results.append(train_final(name, params_by_model[name], data))
    best = max(results, key=lambda r: r["val_macro_f1"])

    comparison = pd.DataFrame([
        {**{k: v for k, v in r.items() if k not in ("params", "checkpoint")}, "params": json.dumps(r["params"])}
        for r in results
    ])
    comparison.to_csv(out_path("reports", "model_comparison.csv"), index=False)
    best_path = out_path("models", "best_model.keras")
    shutil.copy(best["checkpoint"], best_path)

    meta = {
        "model_name": best["model"],
        "img_size": list(config.IMG_SIZE),
        "params": best["params"],
        "seed": config.SEED,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "class_names": data.class_names,
        "gradcam_layer": GRADCAM_LAYERS[best["model"]],
        "val_macro_f1": best["val_macro_f1"],
        # Daftar file val disimpan agar run berikutnya bisa mendeteksi jika split berubah.
        "val_files": [str(p) for p in data.val_paths],
    }
    save_json(meta, out_path("models", "meta.json"))

    for r in results:
        marker = "  <- dipilih" if r is best else ""
        info(f"{r['model']:<12}: val macro-F1 = {r['val_macro_f1']:.4f} | val acc = {r['val_accuracy']:.4f}{marker}")
    info(f"Model tersimpan: {best_path}")
    return meta
