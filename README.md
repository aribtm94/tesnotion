# Generalisasi Estimasi Berat & Deteksi Anomali Broiler (YOLO + MOWA)

Pipeline skripsi untuk **mengukur generalisasi** deteksi broiler lintas kamera/dataset dan
**mendeteksi anomali berat** ayam. Alur inti:

1. **Deteksi** broiler dengan YOLO (model terlatih pada dataset PIO).
2. **Rektifikasi fisheye/wide-angle** tiap gambar dengan **MOWA** (model end-to-end, TPAMI 2025)
   sebagai praproses, lalu evaluasi ulang untuk melihat apakah mAP membaik (eksperimen A/B).
3. **Estimasi berat relatif** tiap bbox terhadap standar Cobb500.
4. **Deteksi anomali** berat dengan **voting ensemble unsupervised** (adaptasi paper
   cattle-outlier), dibandingkan dengan metode percentile lama.
5. **Dashboard Streamlit** untuk menelusuri hasil per dataset.

> Model MOWA memakai lisensi **S-Lab 1.0 (non-komersial)** — sesuai untuk riset/skripsi.

---

## 1. Struktur Repo

```
src/                     # Script pipeline (deteksi, eval, anomali) — lihat di bawah
  eval_detection.py      # Evaluasi mAP YOLO pada PIO + external
  compare_ab.py          # Bandingkan baseline vs MOWA (A/B)
  anomaly_ensemble.py    # Voting ensemble anomali (metode baru)
  anomaly_compare.py     # Ensemble vs percentile → rekomendasi
  common.py              # Util bersama (path, Cobb500, statistik, IO)
  ...                    # extract_bbox_features, estimate_weight_anomalies, dll
dashboard/app.py         # Dashboard Streamlit
src/mowa_rectify.py   # Praproses MOWA (rectify + warp label)
vendor/MOWA/             # Repo MOWA (di-clone, TIDAK di-commit)
data/                    # Dataset + hasil rectified (TIDAK di-commit)
train model/             # Bobot YOLO hasil training (TIDAK di-commit)
reports/ , features/     # Output CSV/JSON/HTML
docs/                    # Dokumentasi & catatan riset
```

Folder besar (`data/`, `vendor/`, `train model/`, `wheels/`, `.venv*`) sengaja **di-.gitignore**;
repo fokus pada kerangka kode. Lihat bagian **Prasyarat data & bobot**.

---

## 2. Prasyarat Lingkungan (dua venv terpisah)

Dua virtual-env Python 3.10 dipakai supaya dependensi tidak bentrok:

| venv | untuk | paket kunci |
|------|-------|-------------|
| `.venv-yolo` | eval YOLO, anomali, dashboard | ultralytics, torch cu121, streamlit |
| `.venv-mowa` | inferensi MOWA | torch cu121, timm, einops |

**GPU wajib** (MOWA meng-hardcode `.cuda()`). Diuji pada RTX 4060 Laptop 8 GB, CUDA 12.1.

### 2a. Buat `.venv-yolo`
```bash
python -m venv .venv-yolo
# torch CUDA cu121 (pakai wheel lokal bila ada di wheels/, atau unduh):
.venv-yolo/Scripts/python.exe -m pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
.venv-yolo/Scripts/python.exe -m pip install ultralytics "numpy<2" opencv-python==4.9.0.80 streamlit
```

### 2b. Buat `.venv-mowa` + MOWA
```bash
git clone https://github.com/KangLiao929/MOWA vendor/MOWA
# unduh checkpoint (GDrive id 1fxQbD1TLoRnW8lG2a8KMinmD6Jlol8EX) ->
#   vendor/MOWA/checkpoint/mowa_pretrained.pth
python -m venv .venv-mowa
.venv-mowa/Scripts/python.exe -m pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
.venv-mowa/Scripts/python.exe -m pip install timm==0.9.12 einops "numpy<2" opencv-python==4.9.0.80
```
> **Urutan penting:** pasang torch CUDA lebih dulu sampai selesai, baru `timm` dkk —
> memasang `timm` sebelum torch bisa menarik torch CPU dan menimpa build CUDA.

---

## 3. Prasyarat Data & Bobot (tidak ada di git)

Siapkan struktur berikut (format label YOLO: `class cx cy w h` ternormalisasi):

```
data/images/{train,val}/*.jpg           # dataset PIO
data/labels/{train,val}/*.txt
data/external/broiler_instance_seg/train/{images,labels}
data/external/chicken_detection_fum/{test,valid,train}/{images,labels}
data/FilePrefixCode.xlsx , data/classes.txt
train model/runs_compare/cmp_yolov8m/weights/best.pt   # bobot YOLO terbaik
```

Dataset external bisa diunduh ulang: `python src/download_roboflow_datasets.py --api-key <KEY>`
(butuh API key Roboflow gratis; jangan commit key).

---

## 4. Menjalankan Pipeline (urut)

Semua perintah dari root proyek.

### Langkah 1 — Evaluasi baseline (kondisi A)
```bash
.venv-yolo/Scripts/python.exe src/eval_detection.py \
    --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
    --out reports/eval_baseline.json
```

### Langkah 2 — Rektifikasi MOWA + warp label (semua dataset)
```bash
# PIO val
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/images/val --labels data/labels/val \
    --output data/rectified/pio_val --label-mode warp
# broiler_instance_seg
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/external/broiler_instance_seg/train/images \
    --labels data/external/broiler_instance_seg/train/labels \
    --output data/rectified/broiler_instance_seg --label-mode warp
# chicken_detection_fum (ulangi untuk test/valid/train ke output yang sama)
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/external/chicken_detection_fum/test/images \
    --labels data/external/chicken_detection_fum/test/labels \
    --output data/rectified/chicken_detection_fum --label-mode warp
```
`--label-mode warp` mentransformasi bbox mengikuti rectification (instance-mask + nearest),
jadi label tetap selaras dengan gambar hasil MOWA. ~2–5 dtk/gambar pada RTX 4060.

### Langkah 3 — Evaluasi MOWA (kondisi B) + bandingkan A/B
```bash
.venv-yolo/Scripts/python.exe src/eval_detection.py \
    --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" \
    --rectified-root data/rectified --out reports/eval_mowa.json
.venv-yolo/Scripts/python.exe src/compare_ab.py       # -> reports/ab_comparison.{json,csv,html}
```

### Langkah 3b — (Bila MOWA lebih buruk) Fine-tune "rectify-both"
Bila `compare_ab` memberi verdict **mowa_worse**, itu wajar: detektor dilatih pada gambar
asli lalu diuji pada gambar rectified (domain mismatch). Fix sesuai literatur (KITTI-360
fisheye benchmark; FisheyeYOLO/WoodScape): fine-tune detektor pada domain rectified.
```bash
# rectify juga train set PIO
.venv-mowa/Scripts/python.exe src/mowa_rectify.py \
    --input data/images/train --labels data/labels/train \
    --output data/rectified/pio_train --label-mode warp
# fine-tune YOLOv8m pada rectified train+val
.venv-yolo/Scripts/python.exe src/finetune_rectified.py --epochs 40
# eval ulang bobot hasil fine-tune (kondisi B') + bandingkan
.venv-yolo/Scripts/python.exe src/eval_detection.py \
    --weights "train model/runs_rectified/ft_rectified_yolov8m/weights/best.pt" \
    --rectified-root data/rectified --out reports/eval_mowa_ft.json
.venv-yolo/Scripts/python.exe src/compare_ab.py --mowa reports/eval_mowa_ft.json \
    --out-prefix reports/ab_comparison_ft
```

### Langkah 4 — Fitur berat + anomali
```bash
# fitur & estimasi berat (menghasilkan features/weight_estimates_compare.csv)
.venv-yolo/Scripts/python.exe src/extract_bbox_features.py
.venv-yolo/Scripts/python.exe src/estimate_weight_anomalies.py
.venv-yolo/Scripts/python.exe src/compare_camera_corrections.py
# anomali: ensemble + perbandingan metode
.venv-yolo/Scripts/python.exe src/anomaly_ensemble.py     # -> reports/anomaly_ensemble_*
.venv-yolo/Scripts/python.exe src/anomaly_compare.py      # -> reports/anomaly_method_comparison.*
```

### Langkah 5 — Dashboard
```bash
.venv-yolo/Scripts/python.exe -m streamlit run dashboard/app.py
```
Buka URL yang ditampilkan (default http://localhost:8501). Pilih dataset di sidebar; telusuri
tab **Baseline / MOWA / Anomali / Metrik**.

---

## 5. Output Utama

| File | Isi |
|------|-----|
| `reports/eval_baseline.json` / `eval_mowa.json` | mAP per dataset (A / B) |
| `reports/ab_comparison.html` | Tabel + verdict A/B |
| `reports/anomaly_ensemble_report.html` | Ringkasan voting ensemble |
| `reports/anomaly_method_comparison.html` | Ensemble vs percentile + rekomendasi |
| `reports/anomaly_review_sample.csv` | Sampel bbox skor tertinggi untuk cek mata |

---

## 6. Hasil Eksperimen A/B (ringkas)

Model: YOLOv8m PIO. Metrik primer mAP50-95. Tiga kondisi:

| Dataset | A: baseline | B: MOWA apa adanya | B': MOWA + fine-tune |
|---------|-------------|--------------------|-----------------------|
| PIO val | 0.710 | 0.638 | **0.683** |
| broiler_instance_seg | 0.536 | 0.457 | **0.530** |
| chicken_detection_fum | 0.058 | 0.049 | **0.058** |
| **mean Δ vs A** | — | **−0.053** | **−0.011** |

**Kesimpulan:** MOWA rectification sebagai praproses **tidak mengalahkan baseline**. Diuji apa
adanya, MOWA lebih buruk (−0.053) karena detektor dilatih pada gambar asli (domain mismatch).
Setelah fine-tune "rectify-both", ~79% kerugian pulih (−0.011) tetapi tetap sedikit di bawah
baseline. Ini sejalan dengan literatur (WoodScape/FisheyeYOLO): untuk kamera berdistorsi ringan,
undistortion tidak andal meningkatkan detektor modern. **Hasil negatif yang valid** untuk skripsi.

## 7. Catatan Metodologi

- **A/B adil:** label ikut di-warp (mode `warp`), jadi perubahan mAP mencerminkan efek
  rectification, bukan misalignment label. Box yang ter-warp keluar frame (FOV trim) dibuang.
- **Dataset external beda domain** dari data latih PIO; angka absolut rendah = sinyal
  generalisasi, bukan bug.
- **Anomali tanpa ground-truth:** "terbaik" dinilai dari stabilitas rate & kesepakatan antar
  metode, bukan akurasi terhadap label anomali (yang tidak tersedia).

## 8. Lisensi & Kredit

- MOWA: KangLiao929/MOWA, **S-Lab License 1.0 (non-komersial)**.
- Standar berat: Cobb500 Broiler Performance Supplement 2022.
- Dataset external: Roboflow Universe (lihat `data/external/*/README.roboflow.txt`).

## 9. Catatan Pemeliharaan

README ini diperbarui langsung melalui integrasi GitHub untuk menguji alur commit dan push.
