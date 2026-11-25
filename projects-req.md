# PROJECT_SPEC: Klasifikasi Kematangan dan Kualitas Buah/Biji Kopi dari Citra dengan CNN

Dokumen ini adalah spesifikasi lengkap untuk membangun aplikasi. Baca seluruhnya sebelum menulis kode, lalu bangun sesuai urutan di bagian "Urutan Pengerjaan".

## 1. Konteks

Ini adalah tugas kuliah Machine Learning (kelompok 3-5 orang). Dosen menilai **pemahaman cara kerja**, bukan tampilan. Karena itu:

- **Tidak ada UI.** Aplikasi berjalan murni di terminal (CLI).
- Kode harus **sederhana, mudah dibaca, dan diberi komentar** yang menjelaskan *kenapa* (bukan hanya *apa*), karena mahasiswa harus bisa menjelaskan tiap bagian saat presentasi.
- Hasilnya harus menghasilkan artefak yang bisa dipresentasikan: grafik, metrik, dan tabel prediksi.

Cakupan materi yang harus terpenuhi (minimal 5 poin dari 9 poin dosen):

| Poin | Materi | Diwujudkan lewat |
|---|---|---|
| 1 | ML Pipeline: cleaning, scaling | Validasi gambar, resize, normalisasi, split, augmentasi |
| 2 | EDA & imbalanced data | Distribusi kelas, class weight |
| 7 | Deep Learning: CNN | CNN baseline + transfer learning (MobileNetV2) |
| 8 | Hyperparameter tuning | Optuna |
| 9 | Ekstraksi metrik | Confusion matrix, loss curve, classification report |

## 2. Tujuan Sistem

Mengklasifikasikan **tingkat kematangan/kualitas** satu jenis objek (buah atau biji kopi) dari citra. Daftar kelas **tidak ditulis manual di kode**: kelas diambil otomatis dari nama subfolder di `data/train/`.

Contoh kelas (bergantung dataset): `unripe`, `ripe`, `overripe`, `rotten`.

> **Catatan untuk pengguna (isi sebelum menjalankan):** pilih satu dataset (misalnya Kaggle: Banana Ripeness, Coffee Bean, atau sejenisnya), lalu susun ke format folder pada bagian 4. Waspadai dataset yang berisi foto objek yang sama dari banyak sudut atau rotasi (misalnya Fruits-360) karena berisiko data leakage.

## 3. Tech Stack

- Python 3.10+
- TensorFlow / Keras (model dan training)
- Optuna (hyperparameter tuning)
- scikit-learn (split, metrik, class weight)
- NumPy, Pandas
- Matplotlib dan Seaborn (grafik)
- Pillow (validasi gambar)
- `argparse` (CLI, library standar)

Buat `requirements.txt` dengan versi yang di-pin sesuai hasil instalasi yang berjalan.

## 4. Struktur Folder

```
coffee-ripeness-cnn/
├── main.py                  # entry point, orkestrasi seluruh pipeline
├── config.py                # semua konstanta dan path
├── requirements.txt
├── README.md                # cara menjalankan (bahasa Indonesia)
├── src/
│   ├── __init__.py
│   ├── eda.py               # analisis distribusi kelas, ukuran gambar, duplikat
│   ├── data.py              # validasi, split, dataset loader, augmentasi
│   ├── models.py            # build_baseline_cnn, build_mobilenet
│   ├── tune.py              # Optuna
│   ├── train.py             # training final, callback, plot kurva
│   ├── evaluate.py          # metrik pada val dan test
│   ├── predict.py           # prediksi folder test, simpan CSV
│   └── utils.py             # seed, logging, helper path
├── data/
│   └── train/
│       ├── <kelas_1>/       # gambar
│       ├── <kelas_2>/
│       └── ...
├── test_data/
│   └── image/               # lihat bagian 8 (berlabel atau tidak)
└── outputs/                 # dibuat otomatis
    ├── models/
    ├── figures/
    ├── reports/
    ├── misclassified/
    └── predictions.csv
```

Nama folder `test_data` memakai underscore (bukan spasi) agar aman di terminal.

## 5. Cara Pakai (perilaku CLI)

Pengguna menjalankan **satu perintah**:

```
python main.py
```

Opsi tambahan (semua opsional):

| Flag | Fungsi | Default |
|---|---|---|
| `--retrain` | Paksa tuning dan training ulang walau model sudah ada | off |
| `--test-dir PATH` | Override folder gambar test | `test_data/image` |
| `--skip-tune` | Lewati Optuna, pakai hyperparameter default di config | off |
| `--gradcam` | Simpan visualisasi Grad-CAM untuk gambar yang diprediksi | off |

**Aturan cache:** jika `outputs/models/best_model.keras` dan `outputs/models/meta.json` sudah ada dan `--retrain` tidak diberikan, **lewati tahap 3 dan 4** (tuning dan training), langsung ke evaluasi dan prediksi. Tujuannya agar demo cepat setelah training selesai (training biasanya dilakukan di Google Colab).

## 6. Alur Pipeline (urutan eksekusi di `main.py`)

Tiap tahap mencetak header `[n/6] Nama Tahap` dan ringkasan hasil ke terminal (bahasa Indonesia).

### Tahap 1: EDA (`src/eda.py`)
- Pindai `data/train/`: hitung jumlah gambar per kelas.
- Validasi gambar: coba buka tiap file dengan Pillow; catat dan **lewati file rusak**; hitung dan laporkan.
- Statistik ukuran gambar (min/maks/rata-rata lebar dan tinggi).
- Deteksi **duplikat** dengan hash isi file (MD5); laporkan jumlahnya dan buang duplikat persis dari data training.
- Hitung rasio imbalanced (kelas terbesar / terkecil) dan cetak.
- Simpan:
  - `outputs/figures/class_distribution.png` (bar chart jumlah per kelas)
  - `outputs/figures/sample_grid.png` (beberapa contoh gambar tiap kelas)
  - `outputs/reports/eda_summary.json`

### Tahap 2: Preprocessing (`src/data.py`)
- Split **stratified** dari `data/train/` menjadi train dan val (default 85/15) memakai daftar path file, dengan `random_state=SEED`. Split dilakukan **sebelum** augmentasi.
- Resize ke `IMG_SIZE` (default 224x224), normalisasi piksel ke rentang yang sesuai model (baseline: skala 1/255; MobileNetV2: `preprocess_input`).
- Augmentasi **hanya untuk data train**: horizontal flip, rotasi kecil, zoom kecil, perubahan brightness/contrast. Val dan test tidak diaugmentasi.
- Hitung **class weight** (`sklearn.utils.class_weight.compute_class_weight`, mode `balanced`) dan gunakan saat training.
- Simpan `outputs/models/class_names.json` (urutan kelas).
- Gunakan `tf.data` dengan `cache`, `shuffle` (hanya train), dan `prefetch`.

### Tahap 3: Tuning Optuna (`src/tune.py`)
- Study per model, `direction="maximize"` dengan metrik **macro-F1 pada val** (atau val accuracy jika macro-F1 sulit dihitung dalam callback; jelaskan pilihan di komentar).
- Jumlah trial: `OPTUNA_TRIALS` (default 15). Epoch per trial: `TUNE_EPOCHS` (default 8) dengan early stopping, supaya cepat.
- Search space:
  - Baseline CNN: `learning_rate` (log, 1e-4 sampai 1e-2), `dropout` (0.2 sampai 0.6), `base_filters` ({16, 32, 64}), `batch_size` ({16, 32, 64}).
  - MobileNetV2: `learning_rate` (log, 1e-5 sampai 1e-3), `dropout` (0.2 sampai 0.6).
- Simpan `outputs/reports/best_params.json` dan grafik `optuna_history.png` serta `optuna_param_importance.png` (lewati grafik importance jika trial terlalu sedikit untuk dihitung, tanpa error).
- Bersihkan sesi Keras (`tf.keras.backend.clear_session()`) antar trial agar memori tidak bocor.

### Tahap 4: Training final (`src/train.py`)
- Latih **dua model** dengan hyperparameter terbaik:
  1. **Baseline CNN** buatan sendiri: 3-4 blok `Conv2D, BatchNorm, ReLU, MaxPool`, lalu `GlobalAveragePooling2D`, `Dropout`, `Dense(softmax)`.
  2. **MobileNetV2** pretrained (ImageNet), `include_top=False`, base dibekukan, kepala klasifikasi baru. Opsional: fine-tuning singkat (unfreeze beberapa layer terakhir dengan learning rate kecil).
- Callback: `EarlyStopping` (patience `EARLY_STOP_PATIENCE`, restore best weights), `ModelCheckpoint`, `ReduceLROnPlateau`.
- Gunakan class weight dari tahap 2.
- Bandingkan kedua model pada val (macro-F1), pilih yang terbaik, simpan sebagai `outputs/models/best_model.keras`.
- Simpan `outputs/models/meta.json`: nama model terpilih, `img_size`, hyperparameter, seed, tanggal training.
- Simpan:
  - `outputs/figures/loss_curve_<model>.png` (train vs val loss)
  - `outputs/figures/accuracy_curve_<model>.png`
  - `outputs/reports/train_log_<model>.csv`
  - `outputs/reports/model_comparison.csv` (tabel perbandingan kedua model)

### Tahap 5: Evaluasi (`src/evaluate.py`)
- Selalu evaluasi pada **val**. Evaluasi pada **test** hanya jika folder test berlabel (lihat bagian 8).
- Hitung: accuracy, macro-F1, weighted-F1, precision/recall/F1 per kelas.
- Simpan:
  - `outputs/figures/confusion_matrix_val.png` dan `confusion_matrix_test.png` (jika ada), versi ternormalisasi per baris dan versi jumlah mentah.
  - `outputs/reports/classification_report.txt`
  - `outputs/reports/metrics.json`
  - `outputs/misclassified/`: salin gambar yang salah ditebak (dengan nama file memuat label benar dan prediksi), batasi maksimal 50 gambar.
- Cetak ringkasan metrik ke terminal.

### Tahap 6: Prediksi (`src/predict.py`)
- Baca semua gambar (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) dari folder test (rekursif sesuai bagian 8).
- Muat `best_model.keras`, `class_names.json`, dan `meta.json`; terapkan preprocessing yang **sama** dengan saat training.
- Prediksi dalam batch.
- Cetak per gambar ke terminal: `nama_file → kelas (confidence%)`. Jika gambar banyak, cetak maksimal 30 baris lalu ringkasan jumlah per kelas.
- Simpan `outputs/predictions.csv` dengan kolom: `filename`, `predicted_class`, `confidence`, satu kolom probabilitas per kelas (`prob_<kelas>`), dan jika berlabel: `true_class`, `correct`.
- Jika `--gradcam`: simpan heatmap Grad-CAM tiap gambar ke `outputs/figures/gradcam/`. Untuk baseline CNN, gunakan layer konvolusi terakhir.

## 7. Konfigurasi (`config.py`)

Semua nilai berikut didefinisikan di satu tempat, tidak ditulis ulang di modul lain:

```python
SEED = 42
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
BATCH_SIZE = 32            # default; bisa dioverride hasil Optuna
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
OPTUNA_TRIALS = 15
TUNE_EPOCHS = 8
TRAIN_DIR = "data/train"
TEST_DIR = "test_data/image"
OUTPUT_DIR = "outputs"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
```

Fungsi `set_seed()` di `utils.py` harus mengatur seed Python, NumPy, dan TensorFlow, dan dipanggil di awal `main.py`.

## 8. Folder Test: Deteksi Berlabel atau Tidak

`predict.py` dan `evaluate.py` mendeteksi struktur `TEST_DIR` otomatis:

- **Berlabel:** `TEST_DIR` berisi subfolder yang namanya cocok dengan `class_names` (misalnya `image/ripe/`, `image/unripe/`). Maka program menghitung metrik test, confusion matrix, dan kolom `true_class` serta `correct` di CSV.
- **Tidak berlabel:** gambar langsung berada di `TEST_DIR`. Maka program hanya membuat prediksi.
- Jika `TEST_DIR` tidak ada atau kosong: cetak peringatan yang jelas, lewati tahap 6, dan tetap selesaikan tahap lainnya tanpa crash.

Gambar di folder test **tidak boleh** dipakai di training atau tuning.

## 9. Aturan Wajib (anti-kesalahan)

1. **Tidak boleh ada data leakage:** split sebelum augmentasi, augmentasi hanya di train, duplikat dibuang sebelum split, scaling/normalisasi tidak memakai statistik dari val/test.
2. **Reproducible:** seed tetap, konfigurasi dan hyperparameter tersimpan di `meta.json`.
3. **Test set dipakai hanya di tahap evaluasi akhir,** tidak pernah di tuning.
4. **Error handling:** file gambar rusak dilewati dengan peringatan; folder data hilang menghasilkan pesan jelas (bukan stack trace panjang) beserta petunjuk perbaikan.
5. **Output terminal** dalam bahasa Indonesia, ringkas, dengan penomoran tahap `[n/6]`.
6. **Komentar kode** dalam bahasa Indonesia, menjelaskan alasan desain (misalnya kenapa class weight, kenapa early stopping, kenapa transfer learning).
7. Identifier (nama variabel dan fungsi) dalam bahasa Inggris, dengan type hints dan docstring singkat.
8. Hindari abstraksi berlebihan: fungsi sederhana lebih baik daripada hierarki class yang rumit.
9. **Tidak membuat UI** (tidak ada web app, GUI, atau dashboard).
10. Jangan mengunduh dataset otomatis; dataset disiapkan pengguna.

## 10. Contoh Output Terminal yang Diharapkan

```
[1/6] EDA
  Kelas: unripe=820 | ripe=910 | overripe=640 | rotten=430
  File rusak dilewati: 3 | Duplikat dibuang: 12
  Rasio imbalanced: 2.1x
  Grafik: outputs/figures/class_distribution.png

[2/6] Preprocessing
  Train: 2.640 | Val: 466 | Class weight: {...}

[3/6] Tuning Optuna (baseline, 15 trial)
  Trial 15/15 selesai. Terbaik: macro-F1=0.91, lr=0.0007, dropout=0.35

[4/6] Training final
  baseline    : val macro-F1 = 0.91
  mobilenetv2 : val macro-F1 = 0.95  <- dipilih
  Model tersimpan: outputs/models/best_model.keras

[5/6] Evaluasi
  Val   -> Accuracy 0.95 | Macro-F1 0.95
  Test  -> Accuracy 0.94 | Macro-F1 0.93 (data test berlabel)

[6/6] Prediksi (test_data/image)
  img_001.jpg -> ripe      (92.4%)
  img_002.jpg -> unripe    (88.1%)
  Hasil lengkap: outputs/predictions.csv
```

## 11. README.md yang Harus Dibuat

Isi (bahasa Indonesia): deskripsi singkat, kebutuhan sistem, instalasi (`pip install -r requirements.txt`), format folder dataset, cara menjalankan (`python main.py` dan flag), penjelasan tiap file output, cara menjalankan di Google Colab (mount Drive, `!python main.py`), dan batasan sistem.

## 12. Urutan Pengerjaan

1. Buat struktur folder, `config.py`, `utils.py` (seed, helper), dan `requirements.txt`.
2. `data.py` (validasi, hash duplikat, split stratified, `tf.data`, augmentasi, class weight) dan `eda.py`.
3. `models.py` (baseline CNN dan MobileNetV2).
4. `train.py` (training, callback, plot kurva, pemilihan model terbaik, simpan `meta.json`).
5. `evaluate.py` (metrik, confusion matrix, misclassified).
6. `predict.py` (deteksi folder berlabel, CSV, opsional Grad-CAM).
7. `tune.py` (Optuna), lalu sambungkan ke `train.py`.
8. `main.py` (orkestrasi, argparse, aturan cache, ringkasan terminal).
9. Uji end-to-end dengan dataset kecil (misalnya 20 gambar per kelas, 2 epoch, 2 trial) untuk memastikan seluruh alur berjalan, lalu tulis `README.md`.

## 13. Kriteria Selesai (Acceptance Criteria)

- [ ] `python main.py` berjalan dari awal sampai akhir tanpa error pada dataset uji kecil.
- [ ] Jalankan kedua kali tanpa `--retrain`: tahap 3-4 dilewati dan program selesai dalam hitungan detik.
- [ ] `--retrain` melatih ulang dan menimpa model.
- [ ] Semua file output pada bagian 6 terbentuk di `outputs/`.
- [ ] Folder test berlabel dan tidak berlabel sama-sama ditangani; folder test kosong tidak membuat program crash.
- [ ] Tidak ada data leakage sesuai aturan bagian 9.
- [ ] Kode dikomentari dalam bahasa Indonesia dan README lengkap.
- [ ] Tidak ada UI.

## 14. Di Luar Cakupan (Non-Goals)

- UI web atau desktop, API server, dan deployment.
- Pengunduhan dataset otomatis.
- Dukungan multi-objek sekaligus (satu proyek = satu jenis objek).
- Deteksi objek (bounding box) atau segmentasi; ini murni klasifikasi citra.