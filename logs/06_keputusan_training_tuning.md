# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 02/11/2026
- **Anggota / Peran:** Ainun Widyah Astanti — Ketua (Prompt Engineer & LLM Navigator) merangkap Anggota 4
- **Target Modul:** Review hasil training penuh & eksperimen hyperparameter tuning (dijalankan Anggota 3), pemilihan konfigurasi terbaik
- **Tautan Sesi Chat:** [https://claude.ai/share/6d62c99b-b497-4275-9a3e-38719a5947e8]
- **Data:** `outputs/metrics_{full30,lr1e3,lr1e4,bs16,bs64}.csv`, `outputs/loss_curve_*.png`, grafik gabungan `outputs/perbandingan_tuning.png`

> Proses teknis menjalankan training & tuning dicatat Anggota 3 (Nur Aisya) di log-nya sendiri. Log ini mencatat **review dan keputusan Ketua** atas hasilnya.

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP PENANGANAN EROR — Hasil "training penuh" yang ternyata 1 epoch

- **Prompt Kelompok:** "katanya sudah training penuh selanjutnya apa"
- **Temuan saat review GitHub:** file yang di-upload belum hasil training penuh.
  ```
  outputs/metrics.csv
  epoch,train_loss,val_loss,val_perplexity,val_accuracy,lr,seconds
  1,3.65244,2.6006,13.4718,0.40899,8.700e-05,338.8
  ```
  - Hanya **1 baris (epoch 1)**, sehingga `loss_curve.png` tampil kosong (satu titik).
  - **338,8 detik/epoch** → training berjalan di **CPU laptop**, bukan GPU Colab (±5 detik/epoch).
  - Checkpoint `outputs/checkpoints/best.pt` (17 MB) ikut di-commit, padahal `*.pt` tidak boleh masuk repo.
- **Analisis:** sebelumnya Anggota 3 menjalankan training 30 epoch di laptop (`5.66s/it`, ±2,7 jam), menghentikannya, lalu meng-upload sisa run tersebut. Kesalahan ini hanya terdeteksi karena isi CSV dan waktu per epoch diperiksa, bukan sekadar melihat bahwa file sudah ada.
- **Solusi:** training diulang di Colab GPU dengan `--run-name full30`; file 1 epoch dan `best.pt` dihapus dari repo; hasil disimpan di `outputs/` dengan akhiran nama run. Folder `outputs/   generated/` (ada spasi di depan nama) dirapikan menjadi `outputs/generated/` dengan `git mv`.
- **Pelajaran:** review hasil eksperimen harus memeriksa **isi** (jumlah epoch, waktu per epoch, device), bukan hanya keberadaan file.

### 2. HASIL TRAINING PENUH (`full30`, lr 3e-4, batch 32, 30 epoch, Colab T4)

| | Epoch 1 | Epoch 30 |
|---|---|---|
| train_loss | 3,649 | 1,229 |
| val_loss | 2,615 | **1,175** |
| Perplexity | 13,67 | **3,24** |
| val_accuracy | 40,9% | **59,3%** |

- Melampaui baseline terbaik tanpa belajar (B, 40,7%) sebesar **+18,6 poin**; melewati 50% sejak epoch 7.
- **Tidak overfitting:** val_loss turun terus sampai epoch 30 dan selalu sedikit di bawah train_loss (train memakai augmentasi transposisi + dropout, validasi tidak).
- **Model belum konvergen:** akurasi masih naik di epoch akhir → dasar untuk eksperimen tuning.

### 3. TAHAP OPTIMASI — Eksperimen Hyperparameter Tuning

- **Prompt Kelompok:** "Eksperimen tuning (lr1e3, lr1e4, bs16, bs64) hasil ini nur aisya taruh dimana" → "coba cek sekarang gimana github"
- **Alasan Khusus:** panduan tugas menetapkan Anggota 3 melakukan "eksperimen fine-tuning (tuning learning rate, batch size)". Satu variabel diubah per run, variabel lain sama dengan `full30`, semua 30 epoch, seed 42.

| Run | Learning rate | Batch size | val_loss terbaik | Perplexity | val_accuracy terbaik | Langkah/epoch |
|---|---|---|---|---|---|---|
| **lr1e3** | **1e-3** | 32 | **1,055** (ep 30) | **2,87** | **63,6%** (ep 29) | 57 |
| bs16 | 3e-4 | 16 | 1,106 (ep 30) | 3,02 | 61,4% (ep 29) | 114 |
| full30 | 3e-4 | 32 | 1,175 (ep 30) | 3,24 | 59,3% (ep 30) | 57 |
| bs64 | 3e-4 | 64 | 1,246 (ep 29) | 3,48 | 55,6% (ep 28) | 29 |
| lr1e4 | 1e-4 | 32 | 1,293 (ep 29) | 3,65 | 53,8% (ep 29) | 57 |
| *Baseline B (tanpa belajar)* | – | – | – | – | *40,7%* | – |

Grafik gabungan: `outputs/perbandingan_tuning.png`.

**Analisis:**
1. **Learning rate paling berpengaruh.** 1e-3 > 3e-4 > 1e-4 secara konsisten di semua epoch. Selisih terbaik-terburuk: 9,8 poin akurasi.
2. **Batch size kecil lebih baik** (16 > 32 > 64). Dengan jumlah epoch sama, batch kecil memberi lebih banyak langkah update (114 vs 57 vs 29 per epoch).
3. Kedua pola menunjukkan hal yang sama: dalam 30 epoch model **dibatasi oleh jumlah/besar update (under-training), bukan overfitting**. Semua run masih membaik di epoch terakhir dan tidak ada yang val_loss-nya naik.
4. Semua konfigurasi jauh di atas baseline B; konfigurasi terbaik **+22,9 poin**.
5. Waktu per epoch hampir sama (5,3–6,1 detik), jadi tidak ada trade-off kecepatan yang berarti.

### 4. KEPUTUSAN KETUA

1. **Konfigurasi terbaik: `lr1e3` (learning rate 1e-3, batch size 32)** — val_loss 1,055, perplexity 2,87, val_accuracy 63,6%. Model ini dipakai untuk sampel musik final dan presentasi.
2. **Sampel final** di-generate ulang dari `outputs/checkpoints/best_lr1e3.pt` (sampel di `outputs/generated/` saat ini berasal dari `full30`). Disarankan beberapa temperature (0,7 / 0,9 / 1,2) untuk perbandingan.
3. `config.py` **tidak diubah** (nilai default tetap 3e-4/32 agar riwayat eksperimen tetap jelas); konfigurasi terbaik dijalankan lewat argumen `--lr 1e-3`.
4. **Opsional (jika ada waktu):** karena model belum konvergen, eksperimen lanjutan yang paling menjanjikan adalah `--lr 1e-3 --batch-size 16` dan/atau `--epochs 60`.
5. Checkpoint `.pt` disimpan di Google Drive kelompok, tidak di-commit.
