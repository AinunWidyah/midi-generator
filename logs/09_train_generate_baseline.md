# Log Interaksi LLM - Kelompok MIDI

| | |
|---|---|
| **Anggota** | Nur Aisya - Anggota 3 (Training Loop, Evaluasi, dan Generasi) |
| **LLM** | Claude (claude.ai) |
| **Tautan Sesi Chat** | https://claude.ai/share/0cbb158f-1c76-4209-95dd-3705f99f7eca (berlaku untuk semua sesi di bawah) |

> **Catatan:** kode dari LLM belum pernah dijalankan oleh LLM karena PyTorch tidak tersedia di lingkungannya; LLM hanya memeriksa sintaks. Tanggal bertanda "(cek ulang)" adalah perkiraan dan perlu dicocokkan sebelum disetor.

---

## Sesi 1 - Perencanaan dan pemilihan arsitektur

| | |
|---|---|
| **Tanggal Sesi** | 21/10/2026 (cek ulang) |
| **Anggota / Peran** | Nur Aisya - Anggota 3 |
| **Target Modul** | Pemilihan arsitektur, pembagian peran, draf tokenizer |
| **Tautan Sesi Chat** | https://claude.ai/share/0cbb158f-1c76-4209-95dd-3705f99f7eca |

### Kronologi Prompt & Respons

#### 1. Tahap Inisiasi (Meminta Kode Awal)

**Prompt Kelompok:**

1. "bantu saya mengerjakan tugas besar dalam mata kuliah generative AI ini dengan topik yang saya pilih yaitu Generative Audio (MIDI String Generator): Model mengenerate rangkaian nada musik tanpa pemrosesan berkas biner .wav yang berat." (PDF panduan proyek dilampirkan)
2. "kerjakan secara bertahap"
3. "saya cuma memiliki 3 anggota, sesuaikan saja"
4. "lanjut"

**Alasan Khusus:** PDF panduan dilampirkan agar LLM mengikuti aturan dan rubrik (dilarang menyalin tutorial, wajib dokumentasi). Pengerjaan diminta bertahap agar tiap modul bisa diverifikasi dan dicatat di log. Pembagian peran disesuaikan karena kelompok hanya beranggotakan 3 orang.

**Output Model:** Rekomendasi Transformer decoder-only (GPT minimalis) dengan tokenisasi MIDI berbasis event; alur 7 tahap; pembagian 3 peran; struktur repo; system prompt untuk LLM; draf `tokenizer.py` (kosakata 391 token: `NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`, `VELOCITY`).

#### 2. Tahap Penanganan Eror / Debugging

- **Pesan Eror dari Notebook:** Tidak ada; draf belum dijalankan.
- **Prompt Perbaikan ke LLM:** -
- **Analisis & Solusi Kelompok:** Antarmuka tokenizer kelompok ternyata berbeda dari draf LLM (kelompok memakai `list[str]` dan `TOKEN_TO_ID`), sehingga draf tokenizer tidak dipakai langsung. Yang relevan bagi modul saya adalah arah arsitektur dan alur pengerjaan bertahap.

#### 3. Tahap Optimasi (opsional)

- **Prompt:** "saya cuma memiliki 3 anggota, sesuaikan saja"
- **Hasil / Perubahan Metrik:** Peran Documentarian dipecah dan digabung ke dua anggota lain; setiap anggota mengisi log modulnya sendiri.

---

## Sesi 2 - Kerangka `train.py` dan `generate.py`

| | |
|---|---|
| **Tanggal Sesi** | 26/10/2026 |
| **Anggota / Peran** | Nur Aisya - Anggota 3 |
| **Target Modul** | `src/train.py`, `src/generate.py` |
| **Tautan Sesi Chat** | https://claude.ai/share/0cbb158f-1c76-4209-95dd-3705f99f7eca |

### Kronologi Prompt & Respons

#### 1. Tahap Inisiasi (Meminta Kode Awal)

**Prompt Kelompok:**

1. "ini adalah tugas bagian saya, bantu saya membuat generate.py dan train.py nya" (dua tangkapan layar spesifikasi tim dilampirkan: alur data dan kontrak `train.py`/`generate.py`)
2. "Lanjutkan"

**Alasan Khusus:** Spesifikasi tim dilampirkan agar LLM mengikuti kontrak yang sudah disepakati (bentuk tensor `(B, T)` dan `(B, T, V)`, format checkpoint, signature `generate(...)`), bukan merancang antarmuka sendiri.

**Output Model:**

- `train.py`: loss cross-entropy dengan `.reshape` (bukan `.view`), metrik per epoch (`train_loss`, `val_loss`, `val_perplexity`, `val_accuracy`), `metrics.csv`, `loss_curve.png`, checkpoint berformat dict sesuai spesifikasi.
- `generate.py`: `generate(model, prompt_ids, max_new_tokens, temperature, top_k)` dengan konteks dipotong ke `block_size`, berhenti di `EOS_ID`, hasil disimpan ke `outputs/generated/*.mid`.
- LLM mengasumsikan modul lain yang belum dilihatnya (kelas `MiniGPT`, batch berbentuk tuple `(x, y)`, `VOCAB_SIZE`) dan menandainya `SESUAIKAN`.

#### 2. Tahap Penanganan Eror / Debugging

- **Pesan Eror dari Notebook:** Tidak ada; belum dijalankan.
- **Prompt Perbaikan ke LLM:** -
- **Analisis & Solusi Kelompok:** Ketidakcocokan baru terlihat setelah kode `dataset.py` dibagikan (Sesi 3).

#### 3. Tahap Optimasi (opsional)

- **Prompt:** (bagian dari prompt 1)
- **Hasil / Perubahan Metrik:** Fitur tambahan di luar spesifikasi: AdamW dengan warmup + cosine decay, gradient clipping, early stopping, `--run-name` untuk eksperimen tuning tanpa menimpa hasil, dan `--smoke` untuk menguji loop dengan data acak.

---

## Sesi 3 - Menyesuaikan dengan `dataset.py` kelompok

| | |
|---|---|
| **Tanggal Sesi** | 28/10/2026 (cek ulang) |
| **Anggota / Peran** | Nur Aisya - Anggota 3 |
| **Target Modul** | `src/train.py`, `src/generate.py` (revisi) |
| **Tautan Sesi Chat** | https://claude.ai/share/0cbb158f-1c76-4209-95dd-3705f99f7eca |

### Kronologi Prompt & Respons

#### 1. Tahap Inisiasi (Meminta Kode Awal)

**Prompt Kelompok:** "ini adalah code dari dataset yang kelompok saya gunakan, apakah ada revisi dari code generate.py atau train.py nya?" (kode `dataset.py` dilampirkan utuh)

**Alasan Khusus:** Sesi 2 memakai banyak asumsi. Menempelkan kode asli teman satu tim lebih aman daripada menebak antarmukanya.

**Output Model:** LLM mengidentifikasi ketidakcocokan dan membuat ulang kedua file.

#### 2. Tahap Penanganan Eror / Debugging

**Pesan Eror dari Notebook:** Tidak ada pesan konsol; ketidakcocokan ditemukan lewat analisis kode (belum dijalankan).

| No | Ketidakcocokan | Dampak jika dibiarkan | Perbaikan |
|---|---|---|---|
| 1 | `dataset.py` mengembalikan batch dict (`input_ids`, `labels`, `attention_mask`), kode mengasumsikan tuple `(x, y)` | Gagal membaca batch | Akses lewat `batch["input_ids"]` dan `batch["labels"]` |
| 2 | Label padding bernilai `-100`, loss memakai `ignore_index=PAD_ID` | Nilai `-100` dianggap target di luar jangkauan, loss error | `IGNORE_INDEX = -100` di loss dan evaluasi |
| 3 | `labels` sudah digeser satu token di dataset | Risiko menggeser dua kali | Loss langsung membandingkan `logits[:, t]` dengan `labels[:, t]` |
| 4 | `create_dataloaders` dan `config.py` sudah tersedia | Duplikasi kode | `train.py` memakai `create_dataloaders(config, model_config)` |

- **Prompt Perbaikan ke LLM:** (pertanyaan pada tahap inisiasi sekaligus meminta perbaikan)
- **Analisis & Solusi Kelompok:** LLM tidak bisa tahu antarmuka modul yang belum dilihatnya. **Pelajaran:** bagikan kontrak atau kode modul terkait sebelum meminta kode yang bergantung padanya.

#### 3. Tahap Optimasi (opsional)

- **Prompt:** -
- **Hasil / Perubahan Metrik:** Penanganan data validasi kosong (checkpoint terbaik dipilih dari `train_loss` disertai peringatan), checkpoint hanya berisi tipe dasar agar aman dimuat, dan `outputs/` dipetakan relatif ke root proyek.

---

## Sesi 4 - Temuan review B2 dan B3

| | |
|---|---|
| **Tanggal Sesi** | 02/11/2026 |
| **Anggota / Peran** | Nur Aisya - Anggota 3 |
| **Target Modul** | `src/train.py`, `src/generate.py` |
| **Tautan Sesi Chat** | https://claude.ai/share/0cbb158f-1c76-4209-95dd-3705f99f7eca |

### Kronologi Prompt & Respons

#### 1. Tahap Inisiasi (Meminta Kode Awal)

**Prompt Kelompok:** "ada error ini" (tangkapan layar temuan review B2 dan B3 dilampirkan)

**Alasan Khusus:** Temuan review sudah menyebut lokasi, penyebab, dan usulan perbaikan, sehingga cukup dilampirkan agar LLM langsung menerapkannya pada kedua file.

**Output Model:** Revisi `train.py` dan `generate.py`.

#### 2. Tahap Penanganan Eror / Debugging

**Pesan Eror dari Notebook:** Tidak ada pesan konsol; kedua temuan berasal dari code review.

- **B2:** `train.py` dan `generate.py` mengimpor kelas `MiniGPT` yang tidak ada; nama kelas di `model.py` adalah `MiniMusicTransformer`.
- **B3:** `generate.py` memanggil `tokenizer.decode(tokens)` lalu `midi.write(...)`, padahal kontrak tokenizer adalah `decode(tokens, out_path) -> None` yang langsung menulis file (akan memicu `TypeError`).

**Prompt Perbaikan ke LLM:** Temuan B2 dan B3 dilampirkan apa adanya.

**Analisis & Solusi Kelompok:**

- **B2:** seluruh import dan pemanggilan diganti ke `MiniMusicTransformer`.
- **B3:** `save_midi` memanggil `tokenizer.decode(tokens, str(path))`, dengan pembuatan folder `outputs/generated/` otomatis.
- **Pelajaran:** dua tebakan LLM pada Sesi 2 (nama kelas dan perilaku `decode`) luput karena `model.py` dan `tokenizer.py` belum dibagikan. Tanda `SESUAIKAN` adalah daftar titik yang harus diverifikasi lebih dulu.

#### 3. Tahap Optimasi (opsional)

- **Prompt:** -
- **Hasil / Perubahan Metrik:** -
