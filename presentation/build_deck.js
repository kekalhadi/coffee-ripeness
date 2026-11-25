// Generator slide presentasi dari hasil pipeline di outputs/.
// Pemakaian: node presentation/build_deck.js [folder_outputs] [file.pptx]
// Jalankan ulang setelah `python main.py` agar grafik hasil memakai angka training terbaru.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT_DIR = path.resolve(process.argv[2] || path.join(ROOT, "outputs"));
const DECK = path.resolve(process.argv[3] || path.join(__dirname, "presentasi.pptx"));

// --- Palet: espresso (dominan), merah ceri kopi matang (aksen), hijau biji mentah, kuning madu ---
const ESP = "2B1A12", CHERRY = "B3362B", GREEN = "5E8C31", AMBER = "D99A2B", BROWN = "7A5446";
const INK = "2B1A12", MUTED = "6B5B53", CARD = "F3F0EE", GRID = "E3DDD9", WHITE = "FFFFFF", CREAM = "F6E7D8";
const HEAD = "Cambria", BODY = "Calibri";
const MODELS = ["baseline", "mobilenetv2"];
const DISPLAY = { baseline: "CNN Baseline", mobilenetv2: "MobileNetV2" };
const RERUN = "Belum ada hasil: jalankan python main.py, lalu node presentation/build_deck.js";

// ---------- Baca hasil pipeline (semua opsional; slide memakai placeholder jika belum ada) ----------
const out = (...parts) => path.join(OUT_DIR, ...parts);
const readJson = (f) => (fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null);

function parseCsv(file) {
  if (!fs.existsSync(file)) return null;
  const splitLine = (line) => {
    const cells = [];
    let cur = "", quoted = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (quoted && c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') quoted = !quoted;
      else if (c === "," && !quoted) { cells.push(cur); cur = ""; }
      else cur += c;
    }
    return [...cells, cur];
  };
  const [head, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/).map(splitLine);
  return rows.map((r) => Object.fromEntries(r.map((v, i) => [head[i], v])));
}

const eda = readJson(out("reports", "eda_summary.json"));
const meta = readJson(out("models", "meta.json"));
const metrics = readJson(out("reports", "metrics.json"));
const bestParams = readJson(out("reports", "best_params.json"));
const comparison = parseCsv(out("reports", "model_comparison.csv"));
const logs = Object.fromEntries(MODELS.map((m) => [m, parseCsv(out("reports", `train_log_${m}.csv`))]));
const classNames = meta?.class_names || (eda ? Object.keys(eda.class_counts) : null);
const imgSize = meta?.img_size || [224, 224];
const rowOf = (model) => comparison?.find((r) => r.model === model);
const paramsOf = (model) => (rowOf(model) ? JSON.parse(rowOf(model).params) : bestParams?.[model]);
const baseFilters = paramsOf("baseline")?.base_filters || 32;
const fmt = (n, digits = 0) => Number(n).toLocaleString("id-ID", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const bestRow = comparison?.reduce((a, b) => (Number(b.val_macro_f1) > Number(a.val_macro_f1) ? b : a));

// Jumlah train/val per kelas dihitung dari daftar file val di meta.json (split yang benar-benar dipakai).
function splitCounts() {
  if (!eda || !meta?.val_files) return null;
  const val = Object.fromEntries(classNames.map((c) => [c, 0]));
  for (const f of meta.val_files) {
    // shortcut: kelas = folder induk terdekat yang namanya kelas; salah jika subfolder di dalam kelas bernama sama dengan kelas lain.
    const cls = f.split(/[\\/]/).slice(0, -1).reverse().find((s) => s in val);
    if (cls) val[cls] += 1;
  }
  return classNames.map((c) => ({ cls: c, val: val[c], train: eda.class_counts[c] - val[c] }));
}

// ---------- Helper gambar dan bentuk ----------
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 inci
pres.title = "Klasifikasi Kematangan Kopi dengan CNN";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "KONTEN",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.65, fontFace: HEAD, fontSize: 26, bold: true, color: ESP, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Klasifikasi Kematangan Kopi · CNN", options: { x: 0.5, y: 5.2, w: 5, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, margin: 0 } } },
  ],
  slideNumber: { x: 8.9, y: 5.2, w: 0.6, h: 0.3, fontFace: BODY, fontSize: 10, color: MUTED, align: "right" },
});
pres.defineSlideMaster({ title: "GELAP", background: { color: ESP } });

let currentSection = null;
function newSlide(section, title, master = "KONTEN") {
  if (section !== currentSection) { pres.addSection({ title: section }); currentSection = section; }
  const slide = pres.addSlide({ masterName: master, sectionTitle: section });
  if (title) slide.addText(title, { placeholder: "title" });
  return slide;
}

const text = (slide, value, opts) =>
  slide.addText(value, { isTextBox: true, fontFace: BODY, fontSize: 14, color: INK, valign: "top", margin: 0, ...opts });

function bullets(slide, items, opts) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1 } }));
  text(slide, runs, { paraSpaceAfter: 6, ...opts });
}

function card(slide, x, y, w, h, color = CARD) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color }, line: { color, width: 0 } });
}

// Motif deck: lingkaran bernomor berwarna ceri kopi.
function badge(slide, label, x, y, d = 0.42, color = CHERRY) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color, width: 0 } });
  text(slide, String(label), { x, y, w: d, h: d, align: "center", valign: "middle", bold: true, color: WHITE, fontSize: d > 0.4 ? 14 : 11 });
}

function placeholderBox(slide, x, y, w, h, label) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: CARD }, line: { color: MUTED, width: 1, dashType: "dash" } });
  text(slide, [{ text: label, options: { bold: true, breakLine: true } }, { text: RERUN }], { x: x + 0.2, y, w: w - 0.4, h, align: "center", valign: "middle", fontSize: 12, color: MUTED });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return b.toString("ascii", 1, 4) === "PNG" ? { w: b.readUInt32BE(16), h: b.readUInt32BE(20) } : null;
}

// Gambar dipasang "contain" di dalam kotak (rasio aspek dijaga), atau placeholder jika belum ada.
function picture(slide, file, x, y, w, h, label) {
  const size = file && fs.existsSync(file) && pngSize(file);
  if (!size) return placeholderBox(slide, x, y, w, h, label);
  const r = Math.min(w / size.w, h / size.h);
  slide.addImage({ path: file, x: x + (w - size.w * r) / 2, y: y + (h - size.h * r) / 2, w: size.w * r, h: size.h * r, altText: label });
}

// Opsi grafik bersama agar semua grafik tampil konsisten.
const chartBase = (title) => ({
  showTitle: true, title, titleFontFace: BODY, titleFontSize: 13, titleColor: INK,
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
  dataLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelColor: INK,
  legendFontFace: BODY, legendFontSize: 11, legendColor: MUTED, legendPos: "b",
});

function flowArrow(slide, x, y) {
  slide.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x, y, w: 0.18, h: 0.16, rotate: 90, fill: { color: MUTED }, line: { color: MUTED, width: 0 } });
}

const tableHeader = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: ESP } } });
const tableOpts = { fontFace: BODY, fontSize: 12, color: INK, rowH: 0.4, border: { type: "solid", pt: 0.5, color: GRID } };

function darkCover(s) {
  s.addShape(pres.shapes.OVAL, { x: 7.1, y: 0.9, w: 3.6, h: 3.6, fill: { color: CHERRY, transparency: 15 }, line: { color: CHERRY, width: 0 } });
  s.addShape(pres.shapes.OVAL, { x: 6.4, y: 2.9, w: 2.2, h: 2.2, fill: { color: GREEN, transparency: 25 }, line: { color: GREEN, width: 0 } });
}

// =====================================================================
// 1. Judul
// =====================================================================
{
  const s = newSlide("Pembuka", null, "GELAP");
  darkCover(s);
  text(s, "TUGAS MACHINE LEARNING", { x: 0.6, y: 1.0, w: 6, h: 0.35, fontSize: 12, bold: true, color: AMBER, charSpacing: 3 });
  text(s, "Klasifikasi Kematangan Kopi dengan CNN", { x: 0.6, y: 1.45, w: 6.2, h: 1.5, fontFace: HEAD, fontSize: 36, bold: true, color: WHITE });
  text(s, "CNN Baseline vs Transfer Learning MobileNetV2 dalam pipeline CLI 6 tahap", { x: 0.6, y: 3.0, w: 6, h: 0.7, fontSize: 16, color: CREAM });
  if (classNames) text(s, `Kelas: ${classNames.join(" · ")}`, { x: 0.6, y: 3.75, w: 6, h: 0.35, fontSize: 13, color: CREAM });
  text(s, "[Nama Anggota] · [NIM]", { x: 0.6, y: 4.6, w: 6, h: 0.35, fontSize: 13, color: AMBER });
  s.addNotes("Perkenalkan anggota kelompok dan topik: klasifikasi tingkat kematangan buah kopi dari foto menggunakan CNN.");
}

// =====================================================================
// 2. Agenda
// =====================================================================
{
  const s = newSlide("Pembuka", "Agenda");
  const items = ["Latar belakang", "Dataset & EDA", "Pipeline & preprocessing", "Arsitektur model", "Tuning & training", "Hasil & evaluasi", "Kesimpulan"];
  items.forEach((item, i) => {
    const col = i < 4 ? 0 : 1, row = i < 4 ? i : i - 4;
    const x = 0.7 + col * 4.5, y = 1.3 + row * 0.9;
    badge(s, i + 1, x, y, 0.55);
    text(s, item, { x: x + 0.75, y, w: 3.5, h: 0.55, valign: "middle", fontSize: 18 });
  });
  s.addNotes("Porsi terbesar presentasi ada di bagian teknis: pipeline, arsitektur, tuning, dan evaluasi.");
}

// =====================================================================
// 3. Latar belakang (singkat)
// =====================================================================
{
  const s = newSlide("Pembuka", "Latar belakang & tujuan");
  const rows = [
    ["Masalah", "Sortir kematangan kopi masih manual: lambat, subjektif, dan tidak konsisten antarpekerja."],
    ["Pendekatan", "Klasifikasi citra dengan CNN: model belajar ciri warna dan tekstur langsung dari foto."],
    ["Tujuan", "Membandingkan CNN dari nol dengan transfer learning MobileNetV2 secara adil dan reproducible."],
  ];
  rows.forEach(([head, desc], i) => {
    const y = 1.25 + i * 1.25;
    badge(s, i + 1, 0.5, y, 0.5);
    text(s, head, { x: 1.2, y, w: 4.6, h: 0.35, bold: true, fontSize: 16 });
    text(s, desc, { x: 1.2, y: y + 0.38, w: 4.6, h: 0.75, color: MUTED });
  });
  card(s, 6.3, 1.2, 3.2, 3.75, ESP);
  [["6", "tahap pipeline otomatis"], ["2", "model dibandingkan"], ["1", "perintah: python main.py"]].forEach(([num, label], i) => {
    const y = 1.35 + i * 1.2;
    text(s, num, { x: 6.55, y, w: 0.9, h: 0.9, fontFace: HEAD, fontSize: 44, bold: true, color: AMBER, valign: "middle" });
    text(s, label, { x: 7.45, y, w: 1.9, h: 0.9, fontSize: 14, color: WHITE, valign: "middle" });
  });
  s.addNotes("Cukup singkat. Tekankan bahwa warna kulit buah adalah ciri utama kematangan, sehingga cocok untuk CNN.");
}

// =====================================================================
// 4. Dataset & EDA
// =====================================================================
{
  const s = newSlide("Data", "Dataset & EDA");
  if (eda) {
    s.addChart(pres.charts.BAR, [{ name: "Jumlah gambar", labels: Object.keys(eda.class_counts), values: Object.values(eda.class_counts) }], {
      ...chartBase("Jumlah gambar per kelas (setelah dibersihkan)"), x: 0.5, y: 1.1, w: 5.5, h: 3.8, barDir: "col",
      chartColors: [CHERRY], showValue: true, dataLabelPosition: "outEnd", showLegend: false, valAxisMinVal: 0,
    });
  } else placeholderBox(s, 0.5, 1.1, 5.5, 3.8, "Grafik distribusi kelas");
  const ratio = eda?.imbalance_ratio;
  const stats = [
    ["Total gambar", eda ? fmt(eda.total_images) : "–"],
    ["File rusak dilewati", eda ? fmt(eda.corrupt_files.length) : "–"],
    ["Duplikat (MD5) dibuang", eda ? fmt(eda.duplicates_removed.length) : "–"],
    ["Rasio imbalanced (maks/min)", ratio ? `${fmt(ratio, 2)}×${ratio > 1.5 ? " · pakai class weight" : " · seimbang"}` : "–"],
  ];
  stats.forEach(([label, value], i) => {
    const y = 1.1 + i * 0.99;
    card(s, 6.3, y, 3.2, 0.86);
    text(s, value, { x: 6.5, y: y + 0.08, w: 2.9, h: 0.45, fontSize: 20, bold: true, color: CHERRY });
    text(s, label, { x: 6.5, y: y + 0.5, w: 2.9, h: 0.3, fontSize: 11, color: MUTED });
  });
  if (eda) {
    const { width, height } = eda.image_size;
    text(s, `Ukuran asli rata-rata ${fmt(width.mean)}×${fmt(height.mean)} px (min ${width.min}×${height.min}, maks ${width.max}×${height.max}); semua di-resize ke ${imgSize[1]}×${imgSize[0]}.`,
      { x: 0.5, y: 4.92, w: 5.6, h: 0.25, fontSize: 10, color: MUTED });
  }
  s.addNotes("EDA dijalankan sebelum split. File rusak dideteksi dengan membuka penuh gambar; duplikat persis dideteksi lewat hash MD5 supaya tidak bocor antara train dan val.");
}

// =====================================================================
// 5. Contoh citra
// =====================================================================
{
  const s = newSlide("Data", "Contoh citra per kelas");
  picture(s, out("figures", "sample_grid.png"), 0.5, 1.1, 9, 3.75, "Grid contoh gambar per kelas");
  text(s, "Diambil acak (seed tetap) dari data train; tiap baris satu kelas.", { x: 0.5, y: 4.9, w: 7, h: 0.25, fontSize: 11, color: MUTED });
  s.addNotes("Tunjukkan perbedaan visual antarkelas: warna hijau, merah, hingga gelap. Ini yang akan dipelajari model.");
}

// =====================================================================
// 6. Pipeline 6 tahap
// =====================================================================
{
  const s = newSlide("Metode", "Arsitektur pipeline: 6 tahap");
  const steps = [
    ["EDA", "Hitung kelas, cek file rusak & duplikat", "eda_summary.json"],
    ["Preprocessing", "Split stratified, resize, augmentasi", "class_names.json"],
    ["Tuning", "Optuna TPE, objektif macro-F1 val", "best_params.json"],
    ["Training", "Latih 2 model, pilih F1 val terbaik", "best_model.keras"],
    ["Evaluasi", "Confusion matrix & report val/test", "metrics.json"],
    ["Prediksi", "Prediksi folder test + Grad-CAM", "predictions.csv"],
  ];
  const w = 1.38, gap = 0.144;
  steps.forEach(([name, desc, file], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, 1.15, w, 2.45, i === 2 || i === 3 ? CREAM : CARD);
    badge(s, i + 1, x + 0.12, 1.28);
    text(s, name, { x: x + 0.12, y: 1.8, w: w - 0.24, h: 0.32, bold: true, fontSize: 14 });
    text(s, desc, { x: x + 0.12, y: 2.15, w: w - 0.24, h: 0.9, fontSize: 11, color: MUTED });
    text(s, file, { x: x + 0.12, y: 3.15, w: w - 0.2, h: 0.3, fontSize: 9, color: CHERRY });
  });
  card(s, 0.5, 3.85, 9, 0.6, ESP);
  text(s, "$ python main.py [--retrain] [--skip-tune] [--test-dir PATH] [--gradcam]", { x: 0.7, y: 3.85, w: 8.6, h: 0.6, valign: "middle", fontFace: "Courier New", fontSize: 13, color: CREAM });
  text(s, [
    { text: "Mode cache: ", options: { bold: true } },
    { text: "jika best_model.keras dan meta.json sudah ada, tahap 3–4 (kotak krem) dilewati sehingga demo hanya butuh beberapa detik." },
  ], { x: 0.5, y: 4.58, w: 9, h: 0.5, fontSize: 12, color: MUTED });
  s.addNotes("Seluruh alur dijalankan satu perintah. Tiap tahap mencetak header [n/6] di terminal dan menyimpan artefak di folder outputs/. Kelas atau IMG_SIZE yang berbeda dari model tersimpan menghasilkan error yang jelas.");
}

// =====================================================================
// 7. Preprocessing & pencegahan data leakage
// =====================================================================
{
  const s = newSlide("Metode", "Preprocessing & pencegahan data leakage");
  const steps = [
    ["Validasi", "Tiap gambar dibuka penuh (PIL load); file rusak dilewati"],
    ["Deduplikasi MD5", "Sebelum split, jadi duplikat tidak bisa masuk train dan val"],
    ["Split stratified 85/15", "Seed 42; proporsi kelas di val sama dengan train"],
    ["Class weight balanced", "w_c = N / (K · n_c): minoritas berbobot lebih besar"],
    ["Cek leakage test", "Gambar test identik dengan train → peringatan"],
  ];
  steps.forEach(([head, desc], i) => {
    const y = 1.15 + i * 0.77;
    badge(s, i + 1, 0.5, y + 0.05, 0.4);
    text(s, head, { x: 1.05, y, w: 3.9, h: 0.3, bold: true, fontSize: 14 });
    text(s, desc, { x: 1.05, y: y + 0.3, w: 3.9, h: 0.42, fontSize: 12, color: MUTED });
  });
  const split = splitCounts();
  if (split) {
    s.addChart(pres.charts.BAR, [
      { name: "Train", labels: split.map((r) => r.cls), values: split.map((r) => r.train) },
      { name: "Val", labels: split.map((r) => r.cls), values: split.map((r) => r.val) },
    ], {
      ...chartBase("Hasil split per kelas"), x: 5.2, y: 1.1, w: 4.3, h: 3.95, barDir: "col", barGrouping: "stacked",
      chartColors: [BROWN, AMBER], showValue: true, dataLabelPosition: "ctr", dataLabelColor: WHITE, showLegend: true, valAxisMinVal: 0,
    });
  } else placeholderBox(s, 5.2, 1.1, 4.3, 3.95, "Grafik split train/val");
  s.addNotes("Kunci utama: split dilakukan sebelum augmentasi, dan duplikat dibuang sebelum split. Data test tidak pernah dipakai untuk training maupun tuning.");
}

// =====================================================================
// 8. Input pipeline tf.data & augmentasi
// =====================================================================
{
  const s = newSlide("Metode", "Input pipeline tf.data & augmentasi");
  const chain = ["Load (PIL)", "Resize uint8", "Cache RAM", "Shuffle", "Augmentasi", "Preprocess", "Batch", "Prefetch"];
  const trainOnly = new Set(["Shuffle", "Augmentasi"]);
  const w = 0.93, gap = 0.22;
  chain.forEach((name, i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, 1.15, w, 0.6, trainOnly.has(name) ? AMBER : ESP);
    text(s, name, { x, y: 1.15, w, h: 0.6, align: "center", valign: "middle", fontSize: 11, bold: true, color: WHITE });
    if (i < chain.length - 1) flowArrow(s, x + w + 0.02, 1.37);
  });
  text(s, "Kotak kuning hanya untuk data train. Cache diletakkan sebelum augmentasi, jadi tiap epoch mendapat variasi baru tanpa membaca ulang disk.",
    { x: 0.5, y: 1.85, w: 9, h: 0.45, fontSize: 12, color: MUTED });
  s.addTable([
    [tableHeader("Augmentasi"), tableHeader("Nilai"), tableHeader("Meniru")],
    ["RandomFlip", "horizontal", "arah foto"],
    ["RandomRotation", "±0,05 putaran (≈18°)", "kemiringan kamera"],
    ["RandomZoom", "±10%", "jarak kamera"],
    ["RandomBrightness", "±10%", "pencahayaan"],
    ["RandomContrast", "±10%", "kualitas kamera"],
  ], { ...tableOpts, x: 0.5, y: 2.45, w: 5.3, colW: [1.8, 1.9, 1.6] });
  card(s, 6.1, 2.45, 3.4, 2.45);
  text(s, "Normalisasi per model", { x: 6.3, y: 2.58, w: 3, h: 0.3, bold: true, fontSize: 14 });
  bullets(s, [
    "CNN Baseline: x / 255 → [0, 1]",
    "MobileNetV2: preprocess_input → [−1, 1], sama seperti saat pretraining",
    "Tanpa augmentasi hue/saturasi: warna adalah ciri utama kematangan",
  ], { x: 6.3, y: 2.95, w: 3.05, h: 1.9, fontSize: 12 });
  s.addNotes("Augmentasi sengaja ringan. Perubahan warna tidak dipakai karena akan merusak label: buah merah yang digeser hue-nya bisa terlihat seperti buah mentah.");
}

// =====================================================================
// 9. Model 1: CNN Baseline
// =====================================================================
{
  const s = newSlide("Metode", "Model 1: CNN Baseline dari nol");
  const [H] = imgSize;
  const blocks = [{ label: "Input", ch: 3, res: H }];
  for (let b = 0; b < 4; b++) blocks.push({ label: `Blok ${b + 1}`, ch: baseFilters * 2 ** b, res: H / 2 ** (b + 1) });
  const fx = 1.05, maxH = 1.8, baseY = 3.25;
  blocks.forEach((blk, i) => {
    const h = 0.3 + maxH * (blk.res / H);
    const x = 0.5 + i * fx;
    card(s, x, baseY - h, 0.8, h, i === 0 ? GRID : i === 4 ? CHERRY : BROWN);
    text(s, blk.label, { x: x - 0.1, y: baseY + 0.05, w: 1.0, h: 0.25, align: "center", fontSize: 11, bold: true });
    text(s, `${blk.res}×${blk.res}×${blk.ch}`, { x: x - 0.12, y: baseY + 0.3, w: 1.04, h: 0.25, align: "center", fontSize: 9, color: MUTED });
  });
  const headX = 0.5 + 5 * fx;
  [["GAP", `${baseFilters * 8} fitur`], ["Dropout", `p = ${fmt(paramsOf("baseline")?.dropout ?? 0.3, 2)}`], ["Dense", "softmax"]].forEach(([name, sub], i) => {
    const x = headX + i * 1.02;
    card(s, x, 2.5, 0.88, 0.7, i === 2 ? CHERRY : AMBER);
    text(s, name, { x, y: 2.52, w: 0.88, h: 0.38, align: "center", valign: "middle", fontSize: 12, bold: true, color: WHITE });
    text(s, sub, { x, y: 2.88, w: 0.88, h: 0.28, align: "center", fontSize: 9, color: WHITE });
  });
  text(s, "Tinggi kotak = resolusi peta fitur; jumlah filter dilipatgandakan tiap blok.", { x: headX, y: 1.25, w: 3, h: 0.6, fontSize: 11, color: MUTED });
  const flat = (H / 16) ** 2 * baseFilters * 8;
  const cards = [
    ["Isi tiap blok", "Conv 3×3 (tanpa bias) → BatchNorm → ReLU → MaxPool 2×2"],
    ["GAP, bukan Flatten", `Flatten = ${fmt(flat)} fitur, GAP = ${baseFilters * 8}: jauh lebih hemat parameter`],
    ["Siap Grad-CAM", `Layer last_conv_relu (${H / 8}×${H / 8}) dipakai untuk heatmap`],
  ];
  cards.forEach(([head, desc], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 3.95, 2.9, 1.1);
    text(s, head, { x: x + 0.15, y: 4.03, w: 2.6, h: 0.3, bold: true, fontSize: 13 });
    text(s, desc, { x: x + 0.15, y: 4.35, w: 2.6, h: 0.65, fontSize: 11, color: MUTED });
  });
  s.addNotes("Baseline adalah pembanding: CNN kecil yang dilatih dari nol. use_bias=False karena BatchNorm sudah punya parameter geser beta. GAP mengurangi overfitting pada dataset kecil.");
}

// =====================================================================
// 10. Model 2: MobileNetV2
// =====================================================================
{
  const s = newSlide("Metode", "Model 2: Transfer learning MobileNetV2");
  text(s, "Latar belakang singkat", { x: 0.5, y: 1.1, w: 4.4, h: 0.3, bold: true, fontSize: 15 });
  bullets(s, [
    "Diperkenalkan Sandler dkk. (Google, 2018) untuk perangkat mobile",
    "Depthwise separable convolution: komputasi ≈ 8–9× lebih kecil dari Conv 3×3 biasa",
    "Inverted residual + linear bottleneck: 17 blok, ekspansi kanal ×6",
    "Bobot pretrained ImageNet (±1,4 juta gambar, 1.000 kelas)",
  ], { x: 0.5, y: 1.5, w: 4.4, h: 2.2, fontSize: 13 });
  card(s, 0.5, 3.8, 4.4, 1.25, CREAM);
  text(s, "Strategi di proyek ini", { x: 0.65, y: 3.88, w: 4.1, h: 0.3, bold: true, fontSize: 13 });
  text(s, "Base dibekukan (trainable = False); hanya kepala klasifikasi yang dilatih. BatchNorm base otomatis berjalan dalam mode inferensi.",
    { x: 0.65, y: 4.2, w: 4.1, h: 0.8, fontSize: 12, color: MUTED });
  const flow = [
    ["Input", `${imgSize[0]}×${imgSize[1]}×3 · preprocess_input`, GRID, INK],
    ["MobileNetV2 base (beku)", "≈154 layer · bobot ImageNet", BROWN, WHITE],
    ["out_relu", `${imgSize[0] / 32}×${imgSize[1] / 32}×1280 · layer Grad-CAM`, BROWN, WHITE],
    ["GAP → Dropout", "1280 fitur", AMBER, WHITE],
    ["Dense softmax", `${classNames ? classNames.length : "K"} kelas · satu-satunya bagian yang dilatih`, CHERRY, WHITE],
  ];
  flow.forEach(([name, sub, fill, color], i) => {
    const y = 1.1 + i * 0.8;
    card(s, 5.3, y, 4.2, 0.62, fill);
    text(s, name, { x: 5.45, y: y + 0.05, w: 3.9, h: 0.28, bold: true, fontSize: 13, color });
    text(s, sub, { x: 5.45, y: y + 0.32, w: 3.9, h: 0.25, fontSize: 10, color });
  });
  s.addNotes("Fitur visual umum seperti tepi, tekstur, dan warna sudah dipelajari dari ImageNet. Dengan dataset kecil, memakai ulang fitur ini lebih akurat dan cepat daripada belajar dari nol.");
}

// =====================================================================
// 11. Perbandingan parameter
// =====================================================================
{
  const s = newSlide("Metode", "Kompleksitas model: total vs parameter dilatih");
  // Angka cadangan diukur dari build_model() dengan 3 kelas dan DEFAULT_PARAMS, dipakai sebelum ada hasil training.
  const fallback = { baseline: [390627, 389667], mobilenetv2: [2261827, 3843] };
  const counts = Object.fromEntries(MODELS.map((m) => {
    const r = rowOf(m);
    return [m, r?.total_params ? [Number(r.total_params), Number(r.trainable_params)] : fallback[m]];
  }));
  s.addChart(pres.charts.BAR, [
    { name: "Total parameter", labels: MODELS.map((m) => DISPLAY[m]), values: MODELS.map((m) => counts[m][0]) },
    { name: "Parameter dilatih", labels: MODELS.map((m) => DISPLAY[m]), values: MODELS.map((m) => counts[m][1]) },
  ], {
    ...chartBase("Jumlah parameter (skala log)"), x: 0.5, y: 1.1, w: 5.6, h: 3.8, barDir: "col",
    chartColors: [BROWN, CHERRY], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0",
    valAxisLogScaleBase: 10, valAxisLabelFormatCode: "#,##0", showLegend: true,
  });
  const [mTotal, mTrain] = counts.mobilenetv2;
  const callouts = [
    [`${fmt((mTrain / mTotal) * 100, 2)}%`, "parameter MobileNetV2 yang benar-benar dilatih"],
    [fmt(counts.baseline[1]), "parameter dilatih CNN Baseline (semua dari nol)"],
    [`${fmt(mTotal / counts.baseline[0], 1)}×`, "ukuran total MobileNetV2 dibanding baseline"],
  ];
  callouts.forEach(([value, label], i) => {
    const y = 1.1 + i * 1.32;
    card(s, 6.4, y, 3.1, 1.18);
    text(s, value, { x: 6.6, y: y + 0.1, w: 2.8, h: 0.55, fontFace: HEAD, fontSize: 28, bold: true, color: CHERRY });
    text(s, label, { x: 6.6, y: y + 0.66, w: 2.8, h: 0.45, fontSize: 11, color: MUTED });
  });
  if (!rowOf("mobilenetv2")?.total_params) text(s, "Angka contoh untuk 3 kelas; terisi otomatis setelah training.", { x: 0.5, y: 4.92, w: 5.6, h: 0.25, fontSize: 10, color: MUTED });
  s.addNotes("MobileNetV2 lebih besar secara total, tetapi yang dilatih hanya kepala Dense. Karena itu training-nya cepat dan risiko overfitting kecil.");
}

// =====================================================================
// 12. Tuning Optuna
// =====================================================================
{
  const s = newSlide("Metode", "Hyperparameter tuning dengan Optuna");
  s.addTable([
    [tableHeader("Parameter"), tableHeader("CNN Baseline"), tableHeader("MobileNetV2")],
    ["learning_rate", "1e-4 – 1e-2 (log)", "1e-5 – 1e-3 (log)"],
    ["dropout", "0,2 – 0,6", "0,2 – 0,6"],
    ["base_filters", "{16, 32, 64}", "–"],
    ["batch_size", "{16, 32, 64}", "32 (tetap)"],
  ], { ...tableOpts, x: 0.5, y: 1.1, w: 4.9, colW: [1.5, 1.7, 1.7] });
  const chips = ["Sampler TPE · seed 42", "15 trial × maks 8 epoch", "Objektif: macro-F1 val", "Early stop patience 3"];
  chips.forEach((chip, i) => {
    const x = 0.5 + (i % 2) * 2.5, y = 3.3 + Math.floor(i / 2) * 0.55;
    card(s, x, y, 2.4, 0.45, CREAM);
    text(s, chip, { x, y, w: 2.4, h: 0.45, align: "center", valign: "middle", fontSize: 12, bold: true });
  });
  text(s, "Learning rate dicari di skala log agar tiap orde besaran mendapat peluang sama. Satu study per model.",
    { x: 0.5, y: 4.5, w: 4.9, h: 0.55, fontSize: 11, color: MUTED });
  const best = meta?.model_name || "mobilenetv2";
  picture(s, out("figures", `optuna_history_${best}.png`), 5.7, 1.1, 3.8, 2.75, `Riwayat trial Optuna (${DISPLAY[best]})`);
  const bp = bestParams?.[best];
  if (bp) {
    const desc = Object.entries(bp).map(([k, v]) => `${k} = ${typeof v === "number" && v < 0.01 ? v.toExponential(1) : typeof v === "number" ? fmt(v, Number.isInteger(v) ? 0 : 2) : v}`).join(" · ");
    text(s, [{ text: `Terbaik (${DISPLAY[best]}): `, options: { bold: true } }, { text: desc }], { x: 5.7, y: 3.95, w: 3.8, h: 1.1, fontSize: 11 });
  }
  s.addNotes("TPE (Tree-structured Parzen Estimator) memodelkan area hyperparameter yang menghasilkan skor bagus, lalu mengambil sampel lebih banyak di area itu. Macro-F1 dipakai karena adil untuk kelas yang tidak seimbang. Tiap trial memanggil clear_session() agar memori tidak menumpuk.");
}

// =====================================================================
// 13. Training final
// =====================================================================
{
  const s = newSlide("Metode", "Training final & callback");
  const callbacks = [
    ["EarlyStopping", "patience 5, restore_best_weights"],
    ["ModelCheckpoint", "simpan hanya saat val_loss membaik"],
    ["ReduceLROnPlateau", "LR × 0,5 setelah 2 epoch mandek"],
    ["CSVLogger", "log loss, accuracy, LR per epoch"],
    ["Class weight", "loss diberi bobot per kelas"],
  ];
  callbacks.forEach(([name, desc], i) => {
    const y = 1.15 + i * 0.78;
    badge(s, i + 1, 0.5, y + 0.05, 0.38);
    text(s, name, { x: 1.0, y, w: 2.9, h: 0.3, bold: true, fontSize: 13 });
    text(s, desc, { x: 1.0, y: y + 0.3, w: 2.9, h: 0.4, fontSize: 11, color: MUTED });
  });
  MODELS.forEach((m, i) => {
    const y = 1.05 + i * 2.05;
    const log = logs[m];
    if (!log) return placeholderBox(s, 4.1, y, 5.4, 1.9, `Kurva loss ${DISPLAY[m]}`);
    const labels = log.map((r) => String(Number(r.epoch) + 1));
    s.addChart(pres.charts.LINE, [
      { name: "Loss train", labels, values: log.map((r) => Number(r.loss)) },
      { name: "Loss val", labels, values: log.map((r) => Number(r.val_loss)) },
    ], {
      ...chartBase(`${DISPLAY[m]}: loss per epoch`), x: 4.1, y, w: 5.4, h: 1.95, chartColors: [GREEN, CHERRY],
      lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5, showLegend: true, legendPos: "r", valAxisLabelFormatCode: "0.0",
    });
  });
  s.addNotes("Jarak besar antara loss train dan loss val menandakan overfitting. Checkpoint terbaik (val_loss terendah) yang dipakai untuk evaluasi, bukan bobot epoch terakhir. Learning rate minimum 1e-7.");
}

// =====================================================================
// 14. Perbandingan model
// =====================================================================
{
  const s = newSlide("Hasil", "Hasil: perbandingan model pada data validasi");
  if (comparison) {
    const labels = comparison.map((r) => DISPLAY[r.model] || r.model);
    s.addChart(pres.charts.BAR, [
      { name: "Val accuracy", labels, values: comparison.map((r) => Number(r.val_accuracy)) },
      { name: "Val macro-F1", labels, values: comparison.map((r) => Number(r.val_macro_f1)) },
    ], {
      ...chartBase("Accuracy & macro-F1 validasi"), x: 0.5, y: 1.1, w: 5.6, h: 3.9, barDir: "col", chartColors: [AMBER, CHERRY],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0", showLegend: true,
    });
    card(s, 6.4, 1.1, 3.1, 3.9, ESP);
    text(s, "MODEL TERPILIH", { x: 6.6, y: 1.3, w: 2.7, h: 0.3, fontSize: 11, bold: true, color: AMBER, charSpacing: 2 });
    text(s, DISPLAY[bestRow.model] || bestRow.model, { x: 6.6, y: 1.6, w: 2.7, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: WHITE });
    text(s, fmt(Number(bestRow.val_macro_f1), 3), { x: 6.6, y: 2.2, w: 2.7, h: 0.9, fontFace: HEAD, fontSize: 48, bold: true, color: AMBER });
    text(s, "macro-F1 validasi", { x: 6.6, y: 3.1, w: 2.7, h: 0.3, fontSize: 12, color: CREAM });
    text(s, `${bestRow.epochs_run} epoch · val loss terbaik ${fmt(Number(bestRow.best_val_loss), 3)}`, { x: 6.6, y: 3.6, w: 2.7, h: 0.6, fontSize: 12, color: CREAM });
  } else placeholderBox(s, 0.5, 1.1, 9, 3.9, "Grafik perbandingan model");
  s.addNotes("Model dipilih otomatis berdasarkan macro-F1 validasi tertinggi, lalu disalin menjadi best_model.keras.");
}

// =====================================================================
// 15. Evaluasi per kelas
// =====================================================================
{
  const split = metrics?.test ? "test" : "val";
  const m = metrics?.[split];
  const s = newSlide("Hasil", `Evaluasi per kelas (data ${split === "test" ? "test" : "validasi"})`);
  if (m) {
    [["Accuracy", m.accuracy], ["Macro-F1", m.macro_f1], ["Weighted-F1", m.weighted_f1]].forEach(([label, v], i) => {
      const x = 0.5 + i * 1.5;
      card(s, x, 1.1, 1.38, 0.8);
      text(s, fmt(v, 3), { x: x + 0.12, y: 1.15, w: 1.2, h: 0.42, fontSize: 18, bold: true, color: CHERRY });
      text(s, label, { x: x + 0.12, y: 1.55, w: 1.2, h: 0.28, fontSize: 11, color: MUTED });
    });
    const cls = Object.keys(m.per_class);
    s.addChart(pres.charts.BAR, ["precision", "recall", "f1-score"].map((k) => ({
      name: k === "f1-score" ? "F1" : k[0].toUpperCase() + k.slice(1), labels: cls, values: cls.map((c) => m.per_class[c][k]),
    })), {
      ...chartBase("Precision, recall, F1 per kelas"), x: 0.5, y: 2.05, w: 4.4, h: 3.0, barDir: "col", chartColors: [AMBER, GREEN, CHERRY],
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0", showLegend: true,
    });
    text(s, `${m.n_images} gambar · ${m.n_misclassified} salah prediksi (disimpan di misclassified/${split}/)`, { x: 5.1, y: 4.85, w: 4.4, h: 0.25, fontSize: 10, color: MUTED });
  } else placeholderBox(s, 0.5, 1.1, 4.4, 3.95, "Metrik per kelas");
  picture(s, out("figures", `confusion_matrix_${split}.png`), 5.1, 1.1, 4.4, 3.65, "Confusion matrix");
  s.addNotes("Confusion matrix kiri berisi jumlah, kanan dinormalisasi per baris (recall). Perhatikan kelas mana yang paling sering tertukar, biasanya kelas yang bersebelahan tingkat kematangannya.");
}

// =====================================================================
// 16. Grad-CAM
// =====================================================================
{
  const s = newSlide("Hasil", "Interpretasi: Grad-CAM");
  const dir = out("figures", "gradcam");
  const files = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.endsWith(".png")).sort().slice(0, 3) : [];
  if (files.length) {
    files.forEach((f, i) => {
      const x = 0.5 + i * 3.05;
      picture(s, path.join(dir, f), x, 1.1, 2.9, 2.75, `Grad-CAM ${f}`);
      text(s, f.replace(/\.png$/, ""), { x, y: 3.88, w: 2.9, h: 0.3, fontSize: 10, color: MUTED, align: "center" });
    });
  } else placeholderBox(s, 0.5, 1.1, 9, 3.05, "Heatmap Grad-CAM (jalankan main.py dengan --gradcam)");
  card(s, 0.5, 4.25, 9, 0.85, CREAM);
  text(s, [
    { text: "Cara kerja: ", options: { bold: true } },
    { text: "gradien skor kelas terhadap peta fitur konvolusi terakhir dirata-rata menjadi bobot α per kanal; heatmap = ReLU(Σ α · A). Area merah paling memengaruhi prediksi." },
  ], { x: 0.7, y: 4.25, w: 8.6, h: 0.85, valign: "middle", fontSize: 12 });
  s.addNotes("Grad-CAM membuktikan model melihat buahnya, bukan latar belakang. Jika heatmap fokus ke latar, ada indikasi model belajar shortcut dari dataset.");
}

// =====================================================================
// 17. Rekayasa: reproducibility & keandalan
// =====================================================================
{
  const s = newSlide("Penutup", "Rekayasa: reproducibility & keandalan");
  const items = [
    ["Seed + determinisme", "SEED=42 untuk Python, NumPy, TF; enable_op_determinism() agar hasil identik antar-run"],
    ["Mode cache", "Model tersimpan dipakai ulang, tahap tuning dan training dilewati"],
    ["Deteksi perubahan data", "Daftar file val disimpan; peringatan jika split berubah"],
    ["Validasi input", "Folder hilang atau kelas < 2 → pesan error jelas, exit code 1"],
    ["Folder test fleksibel", "Berlabel (subfolder kelas) atau tidak; dideteksi otomatis"],
    ["Tes otomatis", "pytest: unit test data + tes end-to-end pipeline"],
  ];
  items.forEach(([head, desc], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 3.05, y = 1.15 + row * 2.0;
    card(s, x, y, 2.9, 1.8);
    badge(s, i + 1, x + 0.18, y + 0.18, 0.4);
    text(s, head, { x: x + 0.7, y: y + 0.18, w: 2.1, h: 0.4, valign: "middle", bold: true, fontSize: 13 });
    text(s, desc, { x: x + 0.18, y: y + 0.72, w: 2.55, h: 1.0, fontSize: 12, color: MUTED });
  });
  s.addNotes("Bagian ini menunjukkan sisi software engineering: hasil bisa diulang, aman dari kesalahan input, dan teruji otomatis.");
}

// =====================================================================
// 18. Kesimpulan
// =====================================================================
{
  const s = newSlide("Penutup", "Kesimpulan & pengembangan");
  const other = comparison?.find((r) => r !== bestRow);
  const findings = bestRow ? [
    `${DISPLAY[bestRow.model]} terpilih dengan macro-F1 validasi ${fmt(Number(bestRow.val_macro_f1), 3)}${other ? ` (${DISPLAY[other.model]}: ${fmt(Number(other.val_macro_f1), 3)})` : ""}`,
    ...(metrics?.test ? [`Data test: accuracy ${fmt(metrics.test.accuracy, 3)}, macro-F1 ${fmt(metrics.test.macro_f1, 3)}`] : []),
    "Pipeline bebas data leakage dan reproducible dengan satu perintah",
  ] : [
    "Hasil numerik akan terisi otomatis setelah training",
    "Transfer learning diharapkan unggul pada dataset kecil",
    "Pipeline bebas data leakage dan reproducible dengan satu perintah",
  ];
  card(s, 0.5, 1.1, 4.4, 3.95);
  text(s, "Kesimpulan", { x: 0.7, y: 1.25, w: 4, h: 0.35, bold: true, fontSize: 16, color: CHERRY });
  bullets(s, findings, { x: 0.7, y: 1.7, w: 4.0, h: 3.2, fontSize: 14 });
  card(s, 5.1, 1.1, 4.4, 3.95, ESP);
  text(s, "Keterbatasan & pengembangan", { x: 5.3, y: 1.25, w: 4, h: 0.35, bold: true, fontSize: 16, color: AMBER });
  bullets(s, [
    "Fine-tuning sebagian layer base MobileNetV2",
    "Resize tanpa menjaga rasio aspek → coba padding",
    "Deteksi duplikat mirip (perceptual hash), bukan hanya identik",
    "Tambah data dari kondisi cahaya dan kamera berbeda",
  ], { x: 5.3, y: 1.7, w: 4.0, h: 3.2, fontSize: 14, color: WHITE });
  s.addNotes("Tutup dengan temuan utama, lalu sebutkan keterbatasan secara jujur sebagai arah pengembangan.");
}

// =====================================================================
// 19. Terima kasih
// =====================================================================
{
  const s = newSlide("Penutup", null, "GELAP");
  darkCover(s);
  text(s, "Terima kasih", { x: 0.6, y: 1.7, w: 6, h: 1.0, fontFace: HEAD, fontSize: 44, bold: true, color: WHITE });
  text(s, "Sesi tanya jawab", { x: 0.6, y: 2.75, w: 6, h: 0.5, fontSize: 20, color: CREAM });
  text(s, "$ python main.py --gradcam", { x: 0.6, y: 3.6, w: 6, h: 0.4, fontFace: "Courier New", fontSize: 14, color: AMBER });
  s.addNotes("Siapkan demo langsung di terminal: mode cache membuat seluruh pipeline selesai dalam beberapa detik.");
}

// Entri folder di ZIP dibuang: membuat file gagal dibuka lewat layanan cloud (OneDrive/Drive).
async function save() {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const source = await JSZip.loadAsync(await pres.write({ outputType: "nodebuffer" }));
  // Salin ke ZIP baru: zip.remove(folder) di JSZip ikut menghapus seluruh isi folder.
  const clean = new JSZip();
  for (const entry of Object.values(source.files).filter((e) => !e.dir)) {
    clean.file(entry.name, await entry.async("nodebuffer"), { createFolders: false });
  }
  fs.mkdirSync(path.dirname(DECK), { recursive: true });
  fs.writeFileSync(DECK, await clean.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log(`Presentasi tersimpan: ${DECK}`);
}
save();
