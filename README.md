# 🎵 MIDI String Generator — Vibe Coding Generative AI from Scratch

Proyek kelompok mata kuliah Generative AI: membangun **Mini-Transformer decoder-only (gaya GPT / Music Transformer)** dari nol untuk menghasilkan rangkaian nada musik dalam bentuk token string, lalu mengubahnya kembali menjadi file `.mid`.

Seluruh kode dibangun dengan metode **Vibe Coding** (kolaborasi dengan LLM). Riwayat interaksi didokumentasikan di folder [`logs/`](logs/).

## Alur Sistem

```
File MIDI → Tokenizer → String token → ID angka → Transformer → Token baru → Decoder → File .mid baru
```

## Anggota & Pembagian Tugas

| Peran | Nama | NIM | Modul |
|---|---|---|---|
| 1. Prompt Engineer & LLM Navigator (Ketua) | Ainun Widyah Astanti | 103012300101 | `src/model.py`, integrasi |
| 2. Data Pipeline & Preprocessing Engineer | Atha Imtinan Ukirhayekti Larasati | 103012300183 | `src/tokenizer.py`, `src/dataset.py` |
| 3. Training Loop & Hyperparameter Tuning Engineer | Nur Aisya Jamil | 103012330138 | `src/train.py`, `src/generate.py` |
| 4. Lead Documentarian & Error Handler | All Member | All Member | `logs/`, laporan akhir |

## Struktur Folder

```
midi-generator/
├── data/        # dataset mentah & hasil tokenisasi (tidak di-commit)
├── src/
│   ├── tokenizer.py   # MIDI <-> token string
│   ├── dataset.py     # Dataset & DataLoader
│   ├── model.py       # Mini-Transformer decoder-only
│   ├── train.py       # training loop
│   └── generate.py    # sampling & ekspor .mid
├── notebooks/   # eksperimen di Google Colab
├── outputs/     # hasil .mid, grafik loss, checkpoint
└── logs/        # Log Interaksi LLM
```

## Konfigurasi Awal (disepakati)

- Framework: PyTorch
- Dataset: Nottingham / JSB Chorales
- Model: d_model=256, n_head=4, n_layer=4, block_size=256, dropout=0.1
- Library I/O MIDI: `pretty_midi` (hanya baca/tulis file; tokenizer ditulis sendiri)

## Cara Menjalankan

```bash
pip install -r requirements.txt
# (akan dilengkapi seiring pengerjaan)
```

## Aturan Kerja Tim

1. Setiap anggota bekerja di **branch sendiri**, lalu buat Pull Request ke `main`.
2. Pesan commit memakai format: `[peran] deskripsi singkat`, misalnya `[data] tambah fungsi encode tokenizer`.
3. Setiap sesi dengan LLM wajib dicatat di `logs/` memakai `logs/TEMPLATE_LOG.md`.
4. Dilarang menyalin kode dari tutorial/GitHub orang lain.
