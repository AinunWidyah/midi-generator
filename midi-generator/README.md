# 🎵 MIDI String Generator — Vibe Coding Generative AI from Scratch

Proyek kelompok mata kuliah Generative AI: membangun **Mini-Transformer decoder-only (gaya GPT / Music Transformer)** dari nol untuk menghasilkan rangkaian nada musik dalam bentuk token string, lalu mengubahnya kembali menjadi file `.mid`.

Seluruh kode dibangun dengan metode **Vibe Coding** (kolaborasi dengan LLM). Riwayat interaksi didokumentasikan di folder [`logs/`](logs/).

## Alur Sistem

```
File MIDI → Tokenizer → String token → ID angka → Transformer → Token baru → Decoder → File .mid baru
```

## Anggota & Pembagian Tugas

| Peran | Nama | Modul |
|---|---|---|
| 1. Prompt Engineer & LLM Navigator (Ketua) | _isi nama_ | `src/model.py`, integrasi |
| 2. Data Pipeline & Preprocessing Engineer | _isi nama_ | `src/tokenizer.py`, `src/dataset.py` |
| 3. Training Loop & Hyperparameter Tuning Engineer | _isi nama_ | `src/train.py`, `src/generate.py` |
| 4. Lead Documentarian & Error Handler | _isi nama_ | `logs/`, laporan akhir |

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
