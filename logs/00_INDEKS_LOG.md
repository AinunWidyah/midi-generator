# 📚 Indeks Log Interaksi LLM — MIDI String Generator

Daftar seluruh sesi vibe coding kelompok secara kronologis. Setiap log mengikuti `logs/TEMPLATE_LOG.md` (Inisiasi → Penanganan Eror → Optimasi).

**Pengelola:** Ainun Widyah Astanti (Anggota 4 — Lead Documentarian & Error Handler)
**Terakhir diperbarui:** 02/11/2026

## Anggota & Peran

| Peran | Nama | Akun GitHub | Modul |
|---|---|---|---|
| 1. Prompt Engineer & LLM Navigator (Ketua) | Ainun Widyah Astanti | AinunWidyah | `config.py`, `model.py`, kontrak, integrasi |
| 2. Data Pipeline & Preprocessing Engineer | Laratha | Laratha | `tokenizer.py`, `dataset.py`, `preprocessing/` |
| 3. Training Loop & Hyperparameter Tuning Engineer | Nur Aisya Jamil | sasyalonic | `train.py`, `generate.py` |
| 4. Lead Documentarian & Error Handler | Ainun Widyah Astanti (merangkap) | AinunWidyah | `logs/`, `docs/`, laporan |

## Daftar Sesi

| No | Tanggal | Anggota | Modul / Topik | Eror & debugging | Link chat | File |
|---|---|---|---|---|---|---|
| 01 | 21/10/2026 | Ainun (Ketua) | Perencanaan, jobdesk, setup GitHub, kontrak v1.0 | 4 — folder ganda hasil zip, login browser macet, repo 404, push ditolak | [isi] | [01_setup_proyek.md](01_setup_proyek.md) |
| 02 | 21/10/2026 | Ainun (Ketua) | `model.py` Tahap 1: Causal Self-Attention | 0 | [isi] | [02_model_attention.md](02_model_attention.md) |
| 03 | 21/10/2026 | Ainun (Ketua) | `model.py` Tahap 2–7: Multi-Head Attention s.d. uji overfit | 1 — `view` pada tensor non-contiguous | [isi] | [03_model_lengkap.md](03_model_lengkap.md) |
| 04 | 21/10/2026 | Ainun (Ketua) | Kontrak v1.1–v1.2: lokasi data, `preprocessing.py`, verifikasi asumsi 1 | 3 catatan — skrip memindahkan data mentah, skrip di luar repo, nama file berbeda | [isi] | [04_keputusan_ketua.md](04_keputusan_ketua.md) |
| 05 | 02/11/2026 | Ainun (Ketua + Anggota 4) | Review integrasi, uji end-to-end, verifikasi asumsi 2–3, kontrak v1.3 | 3 — B1 tokenizer vs vocab list, B2 nama kelas `MiniGPT`, B3 salah panggil `decode` | [isi] | [05_uji_integrasi.md](05_uji_integrasi.md) |
| 06 | [isi] | Laratha (Anggota 2) | `tokenizer.py` & `preprocessing/` (audit + filter monofonik) | [isi] | [isi] | ⏳ menunggu |
| 07 | [isi] | Laratha (Anggota 2) | `dataset.py` + perbaikan B1 | [isi] | [isi] | ⏳ menunggu |
| 08 | [isi] | Nur Aisya (Anggota 3) | `train.py` & `generate.py` + perbaikan B2/B3 | [isi] | [isi] | ⏳ menunggu (branch `docs/log-anggota3`) |
| 09 | [isi] | Laratha (Anggota 2) | Augmentasi transposisi (K1) | [isi] | [isi] | ⏳ belum dikerjakan |
| 10 | [isi] | Nur Aisya (Anggota 3) | Training penuh & tuning hyperparameter | [isi] | [isi] | ⏳ belum dikerjakan |

## Dokumen Pendukung

| Dokumen | Isi |
|---|---|
| [`docs/KONTRAK_MODUL.md`](../docs/KONTRAK_MODUL.md) | Kontrak antar-modul (v1.0 → v1.3) |
| [`docs/REVIEW_INTEGRASI_01.md`](../docs/REVIEW_INTEGRASI_01.md) | Review kode seluruh anggota: bug B1–B3, temuan K1–K6, keputusan D1–D5 |
| [`notebooks/uji_integrasi_01.ipynb`](../notebooks/uji_integrasi_01.ipynb) | Bukti eksekusi uji end-to-end di Colab (T4), termasuk audio hasil generate |

## Ringkasan Debugging Kelompok

| Kategori | Jumlah | Contoh |
|---|---|---|
| Git & lingkungan kerja | 4 | Folder ganda hasil ekstrak zip, push ditolak (fetch first) |
| Tensor / PyTorch | 1 | `view` gagal pada tensor hasil slicing → `reshape` |
| Integrasi antar-modul | 3 | Format vocab berbeda, nama kelas berbeda, signature `decode` berbeda |
| Data & kontrak | 3 | Skrip filter memindahkan data mentah, file di luar repo |

## Cara Menambah Log

1. Salin `logs/TEMPLATE_LOG.md` menjadi `logs/NN_topik.md` (NN = nomor urut berikutnya di tabel).
2. Isi ketiga tahap, tempel **pesan eror lengkap** (bukan potongan), dan sertakan **share link chat**.
3. Push lewat branch + Pull Request, lalu kabari Anggota 4 untuk memperbarui indeks ini.
