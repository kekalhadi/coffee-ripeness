"""Arsitektur model: CNN baseline buatan sendiri dan MobileNetV2 (transfer learning)."""

import keras

import config

# Layer peta fitur konvolusi terakhir tiap model, dipakai Grad-CAM.
GRADCAM_LAYERS = {"baseline": "last_conv_relu", "mobilenetv2": "out_relu"}


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    """Compile dengan Adam dan sparse categorical crossentropy (label berupa indeks kelas)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn(n_classes: int, base_filters: int, dropout: float, learning_rate: float) -> keras.Model:
    """CNN 4 blok Conv2D-BatchNorm-ReLU-MaxPool, lalu GAP, Dropout, Dense softmax."""
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
    x = inputs
    n_blocks = 4
    for block in range(n_blocks):
        # Jumlah filter dilipatgandakan tiap blok: resolusi mengecil, fitur makin abstrak
        # (tepi -> tekstur -> bentuk), jadi butuh lebih banyak "detektor" fitur.
        is_last = block == n_blocks - 1
        x = keras.layers.Conv2D(base_filters * 2**block, 3, padding="same", use_bias=False)(x)
        # BatchNorm menstabilkan distribusi aktivasi sehingga training lebih cepat konvergen.
        # use_bias=False karena BatchNorm sudah punya parameter geser (beta).
        # momentum 0.9 (default 0.99): dataset kecil = sedikit step, statistik inferensi harus cepat konvergen.
        x = keras.layers.BatchNormalization(momentum=0.9)(x)
        x = keras.layers.ReLU(name="last_conv_relu" if is_last else None)(x)
        x = keras.layers.MaxPooling2D()(x)
    # GlobalAveragePooling jauh lebih hemat parameter daripada Flatten, sehingga
    # mengurangi overfitting pada dataset kecil.
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(inputs, outputs, name="baseline"), learning_rate)


def build_mobilenet(n_classes: int, dropout: float, learning_rate: float) -> keras.Model:
    """MobileNetV2 pretrained ImageNet dengan base dibekukan dan kepala klasifikasi baru."""
    # Transfer learning: fitur visual umum (tepi, tekstur, warna) sudah dipelajari dari
    # 1,4 juta gambar ImageNet. Dengan dataset kecil, memakai ulang fitur itu jauh lebih
    # akurat dan cepat daripada belajar dari nol.
    base = keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    # Dibekukan agar bobot pretrained tidak rusak oleh gradien besar dari kepala yang
    # masih acak. BatchNorm yang dibekukan otomatis berjalan dalam mode inferensi.
    base.trainable = False
    # Model dibangun "rata" dari base.input (bukan base sebagai satu layer bersarang)
    # agar layer "out_relu" bisa diakses langsung untuk Grad-CAM.
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    return _compile(keras.Model(base.input, outputs, name="mobilenetv2"), learning_rate)


def build_model(model_name: str, n_classes: int, params: dict) -> keras.Model:
    """Bangun model berdasarkan nama dan dict hyperparameter."""
    if model_name == "baseline":
        return build_baseline_cnn(n_classes, params["base_filters"], params["dropout"], params["learning_rate"])
    if model_name == "mobilenetv2":
        return build_mobilenet(n_classes, params["dropout"], params["learning_rate"])
    raise ValueError(f"Model tidak dikenal: {model_name}")
