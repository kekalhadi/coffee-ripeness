"""Tahap 3: hyperparameter tuning dengan Optuna (satu study per model)."""

import warnings

import keras
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

import config
from src.data import PreparedData
from src.train import early_stopping, fit_model, score_on_val
from src.utils import info, out_path, save_json

# Fungsi plot matplotlib Optuna masih berlabel "experimental"; peringatannya hanya mengotori terminal.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Search space tiap model."""
    # Learning rate dicari di skala log: nilai bagus bisa 0.0001 atau 0.01, dan skala log
    # memberi peluang yang sama untuk tiap orde besaran.
    if model_name == "baseline":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "base_filters": trial.suggest_categorical("base_filters", [16, 32, 64]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
    # MobileNetV2: base dibekukan, yang dilatih hanya kepala di atas fitur ImageNet yang sudah
    # matang, sehingga rentang LR lebih kecil dan batch_size dibiarkan tetap.
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.6),
        "batch_size": config.BATCH_SIZE,
    }


def _objective(trial: optuna.Trial, model_name: str, data: PreparedData) -> float:
    """Latih singkat dengan parameter trial, kembalikan macro-F1 pada val."""
    params = suggest_params(trial, model_name)
    # Tiap trial membuat model baru; tanpa clear_session graf lama menumpuk dan memori bocor.
    keras.backend.clear_session()
    model, _ = fit_model(model_name, params, data, config.TUNE_EPOCHS, [early_stopping(config.TUNE_PATIENCE)])
    # Macro-F1 dihitung SETELAH training dengan memprediksi seluruh val (bukan di callback):
    # lebih sederhana dan memakai fungsi yang sama dengan pemilihan model di tahap 4.
    # Hanya val yang dipakai; data test tidak pernah disentuh saat tuning.
    return score_on_val(model, model_name, data, params["batch_size"])["val_macro_f1"]


def _save_plots(study: optuna.Study, model_name: str) -> None:
    """Simpan grafik riwayat optimasi dan (jika bisa dihitung) kepentingan parameter."""
    ax = plot_optimization_history(study)
    ax.figure.savefig(out_path("figures", f"optuna_history_{model_name}.png"), dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    try:
        ax = plot_param_importances(study)
        ax.figure.savefig(out_path("figures", f"optuna_param_importance_{model_name}.png"), dpi=120, bbox_inches="tight")
        plt.close(ax.figure)
    except (RuntimeError, ValueError, ZeroDivisionError) as error:
        # Importance butuh beberapa trial dengan hasil berbeda; dengan trial sedikit ini wajar gagal.
        info(f"  Grafik param importance dilewati (trial terlalu sedikit: {error})")
    finally:
        plt.close("all")


def tune_model(model_name: str, data: PreparedData) -> dict:
    """Jalankan satu study Optuna dan kembalikan hyperparameter terbaik."""
    info(f"{model_name} ({config.OPTUNA_TRIALS} trial, maks {config.TUNE_EPOCHS} epoch/trial)")
    # TPESampler memilih kandidat berikutnya berdasarkan trial yang sudah bagus (lebih efisien
    # daripada grid/random search). Seed membuat urutan kandidat bisa diulang.
    study = optuna.create_study(
        direction="maximize", study_name=model_name, sampler=optuna.samplers.TPESampler(seed=config.SEED)
    )

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        info(f"  Trial {trial.number + 1}/{config.OPTUNA_TRIALS}: macro-F1={trial.value:.4f} "
             f"| terbaik={study.best_value:.4f}")

    study.optimize(lambda t: _objective(t, model_name, data), n_trials=config.OPTUNA_TRIALS, callbacks=[report])
    _save_plots(study, model_name)

    # batch_size MobileNetV2 tidak dituning, jadi diambil dari config agar params tetap lengkap.
    best = {"batch_size": config.BATCH_SIZE, **study.best_params}
    shown = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.items())
    info(f"  Terbaik: macro-F1={study.best_value:.4f}, {shown}")
    return best


def tune_all(data: PreparedData) -> dict[str, dict]:
    """Tuning semua model; simpan best_params.json."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # progres dicetak sendiri lewat callback
    best_params = {name: tune_model(name, data) for name in config.MODEL_NAMES}
    save_json(best_params, out_path("reports", "best_params.json"))
    return best_params
