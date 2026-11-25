# Klasifikasi Kematangan/Kualitas Buah atau Biji Kopi dengan CNN

Program terminal (tanpa UI) yang mengklasifikasikan tingkat kematangan/kualitas dari citra.
Isinya EDA, preprocessing, tuning Optuna, training dua model (CNN baseline dan MobileNetV2),
lalu evaluasi dan prediksi. Daftar kelas diambil otomatis dari nama subfolder di `data/train/`.

| Materi kuliah | Diwujudkan di |
|---|---|
| ML Pipeline (cleaning, scaling) | `src/data.py`: validasi gambar, buang duplikat, resize, normalisasi, split, augmentasi |
| EDA & imbalanced data | `src/eda.py` (distribusi kelas, rasio imbalanced) + class weight di `src/data.py` |
| Deep Learning: CNN | `src/models.py`: CNN baseline + transfer learning MobileNetV2 |
| Hyperparameter tuning | `src/tune.py`: Optuna (TPE), metrik macro-F1 pada val |
| Ekstraksi metrik | `src/evaluate.py`: confusion matrix, classification report; `src/train.py`: kurva loss |

## Kebutuhan sistem

- Python 3.10+ (sudah diuji di Python 3.14, macOS, CPU)
- RAM kira-kira 4 GB untuk ~3.000 gambar (gambar di-cache di memori setelah di-resize)
- Koneksi internet saat training pertama untuk mengunduh bobot MobileNetV2 ImageNet (~9 MB)
- GPU tidak wajib. Tanpa GPU, tuning penuh (2 model × 15 trial) bisa memakan waktu berjam-jam,
  jadi sebaiknya training dilakukan di Google Colab.

## Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Format folder dataset

Siapkan dataset sendiri (program tidak mengunduh dataset otomatis):

```
data/train/
├── unripe/      # nama folder = nama kelas
├── ripe/
└── overripe/
```

Folder test (`test_data/image/`) boleh berlabel atau tidak berlabel:

```
test_data/image/ripe/a.jpg      <- berlabel: subfolder bernama sama dengan kelas -> metrik test dihitung
test_data/image/foto1.jpg       <- tidak berlabel: gambar langsung di folder -> hanya prediksi
```

Format yang didukung: `.jpg .jpeg .png .bmp .webp`. Gambar test tidak pernah dipakai untuk training
atau tuning. Hindari dataset yang berisi foto objek yang sama dari banyak sudut (misalnya Fruits-360),
karena foto yang hampir sama bisa masuk train dan val sekaligus (data leakage).

## Cara menjalankan

```bash
python main.py
```

| Flag | Fungsi |
|---|---|
| `--retrain` | Paksa tuning dan training ulang walaupun model sudah ada |
| `--skip-tune` | Lewati Optuna, pakai `DEFAULT_PARAMS` di `config.py` |
| `--test-dir PATH` | Pakai folder test lain (default `test_data/image`) |
| `--gradcam` | Simpan heatmap Grad-CAM tiap gambar test |

**Cache:** jika `outputs/models/best_model.keras` dan `meta.json` sudah ada, tahap 3-4 dilewati,
sehingga demo hanya butuh beberapa detik. Program memberi error yang jelas jika kelas atau `IMG_SIZE`
berbeda dengan model tersimpan, dan memberi peringatan jika isi `data/train` berubah sejak training
(split val berubah sehingga skor val bisa terlalu optimis).

Semua pengaturan (seed, ukuran gambar, epoch, jumlah trial, dan lain-lain) ada di `config.py`.

## Alur 6 tahap

1. **EDA**: hitung gambar per kelas, lewati file rusak, buang duplikat persis (MD5), statistik ukuran, rasio imbalanced.
2. **Preprocessing**: split stratified 85/15 *sebelum* augmentasi, resize 224×224, normalisasi
   (baseline 1/255, MobileNetV2 `preprocess_input`), augmentasi hanya untuk train, class weight `balanced`.
3. **Tuning Optuna**: satu study per model, memaksimalkan macro-F1 val.
4. **Training final**: kedua model dilatih dengan EarlyStopping, ModelCheckpoint, dan ReduceLROnPlateau.
   Model dengan macro-F1 val tertinggi disimpan sebagai `best_model.keras`.
5. **Evaluasi**: val selalu dievaluasi; test hanya jika berlabel.
6. **Prediksi**: semua gambar test diprediksi dan disimpan ke CSV.

## Penjelasan file output (`outputs/`)

| File | Isi |
|---|---|
| `figures/class_distribution.png` | Bar chart jumlah gambar per kelas |
| `figures/sample_grid.png` | Contoh gambar tiap kelas |
| `figures/optuna_history_<model>.png` | Skor tiap trial Optuna |
| `figures/optuna_param_importance_<model>.png` | Hyperparameter yang paling berpengaruh |
| `figures/loss_curve_<model>.png`, `accuracy_curve_<model>.png` | Train vs val per epoch (cek overfitting) |
| `figures/confusion_matrix_val.png`, `confusion_matrix_test.png` | Confusion matrix (jumlah dan ternormalisasi per baris) |
| `figures/gradcam/` | Heatmap Grad-CAM (dengan `--gradcam`) |
| `models/best_model.keras` | Model terpilih |
| `models/baseline.keras`, `models/mobilenetv2.keras` | Checkpoint terbaik masing-masing model |
| `models/class_names.json` | Urutan kelas (indeks output model) |
| `models/meta.json` | Model terpilih, img_size, hyperparameter, seed, tanggal training, daftar file val |
| `reports/eda_summary.json` | Ringkasan EDA, termasuk daftar file rusak dan duplikat |
| `reports/best_params.json` | Hyperparameter terbaik hasil Optuna |
| `reports/train_log_<model>.csv` | Log loss/accuracy/learning rate per epoch |
| `reports/model_comparison.csv` | Perbandingan kedua model pada val |
| `reports/classification_report.txt` | Precision/recall/F1 per kelas |
| `reports/metrics.json` | Accuracy, macro-F1, weighted-F1, dan metrik per kelas |
| `misclassified/<val|test>/` | Maksimal 50 gambar salah tebak; nama file memuat label benar dan prediksi |
| `predictions.csv` | `filename, predicted_class, confidence, prob_<kelas>...` (+ `true_class, correct` jika berlabel) |

## Menjalankan di Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/coffee-ripeness-cnn
!pip install -q optuna            # TensorFlow dan library lain sudah ada di Colab
!python main.py --retrain
```

Aktifkan GPU lewat *Runtime → Change runtime type → GPU*. Setelah selesai, folder `outputs/`
tersimpan di Drive. Salin folder itu ke laptop, lalu jalankan `python main.py` untuk demo cepat (mode cache).

## Slide presentasi

`presentation/build_deck.js` membuat `presentation/presentasi.pptx` (19 slide) dari isi `outputs/`.
Grafik dan gambar hasil terisi otomatis; slide yang datanya belum ada menampilkan kotak placeholder.
Butuh Node.js 18+.

```bash
cd presentation && npm install && cd ..   # sekali saja (memasang pptxgenjs)
python main.py --gradcam                  # hasilkan outputs/ dengan dataset asli
node presentation/build_deck.js           # buat ulang presentasi dari hasil terbaru
```

Ganti `[Nama Anggota] · [NIM]` di slide judul sebelum presentasi.

## Tes

```bash
python -m pytest -q tests
```

`tests/test_data.py` menguji validasi, duplikat, split, dan deteksi folder test.
`tests/test_pipeline.py` menjalankan seluruh pipeline pada dataset sintetis kecil (~20 detik).

## Batasan sistem

- Satu proyek hanya untuk satu jenis objek, dan setiap gambar diasumsikan berisi satu objek (bukan deteksi objek atau segmentasi).
- Gambar di-resize ke 224×224 tanpa menjaga rasio aspek, jadi objek bisa terlihat sedikit gepeng.
- Duplikat yang dideteksi hanya duplikat persis (isi file identik). Foto yang mirip (di-crop atau di-resize) tidak terdeteksi.
- MobileNetV2 hanya dilatih kepalanya (base dibekukan); fine-tuning layer base tidak dilakukan.
- Hasil reproducible di mesin yang sama (seed + `enable_op_determinism`). Di hardware lain (GPU Colab vs CPU laptop), angkanya bisa sedikit berbeda.
- Grafik Optuna disimpan per model (`optuna_history_<model>.png`) karena ada satu study per model.
