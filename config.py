"""Semua konstanta dan path proyek di satu tempat.

Modul lain membaca nilai lewat `config.NAMA` (bukan `from config import NAMA`)
supaya nilai bisa diubah sekali di sini dan langsung berlaku di semua tahap.
"""

# --- Reproducibility ---
SEED = 42

# --- Data ---
IMG_SIZE = (224, 224)  # (tinggi, lebar); 224 adalah ukuran asli input MobileNetV2 ImageNet
VAL_SPLIT = 0.15
TRAIN_DIR = "data/train"
TEST_DIR = "test_data/image"
OUTPUT_DIR = "outputs"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- Training ---
BATCH_SIZE = 32  # default; bisa dioverride hasil Optuna
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 2  # lebih kecil dari early stopping: turunkan LR dulu sebelum menyerah
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# --- Tuning (Optuna) ---
OPTUNA_TRIALS = 15
TUNE_EPOCHS = 8
TUNE_PATIENCE = 3  # lebih agresif dari training final agar tiap trial cepat

# --- Augmentasi (hanya untuk data train) ---
AUG_ROTATION = 0.05  # fraksi dari 360 derajat, kira-kira +-18 derajat
AUG_ZOOM = 0.1
AUG_BRIGHTNESS = 0.1
AUG_CONTRAST = 0.1

# --- Model yang dibandingkan dan hyperparameter default (dipakai dengan --skip-tune) ---
MODEL_NAMES = ("baseline", "mobilenetv2")
DEFAULT_PARAMS = {
    "baseline": {"learning_rate": 1e-3, "dropout": 0.3, "base_filters": 32, "batch_size": BATCH_SIZE},
    # Base MobileNetV2 dibekukan, jadi hanya kepala yang dilatih; LR default Adam aman di sini.
    "mobilenetv2": {"learning_rate": 1e-3, "dropout": 0.3, "batch_size": BATCH_SIZE},
}

# --- Output ---
SAMPLES_PER_CLASS_IN_GRID = 4
MAX_MISCLASSIFIED = 50
MAX_PRINTED_PREDICTIONS = 30
MAX_PRINTED_WARNINGS = 10
