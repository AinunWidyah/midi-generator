# 🔍 Review Integrasi #1 — MIDI String Generator

- **Tanggal:** 02/11/2026
- **Reviewer:** Ainun Widyah Astanti (Ketua)
- **Commit yang direview:** `e19ce65` (main, 29/10/2026)
- **Acuan:** `docs/KONTRAK_MODUL.md`

## Ringkasan

| Modul | Pemilik | Status |
|---|---|---|
| `config.py`, `model.py` | Ketua | ✅ Sesuai kontrak, lulus uji Tahap 1–7 |
| `tokenizer.py` | Anggota 2 (Laratha) | 🔴 **Crash saat di-import** (Bug B1) |
| `dataset.py` | Anggota 2 (Laratha) | 🟡 Jalan setelah B1 diperbaiki; belum ada augmentasi transposisi |
| `preprocessing/` | Anggota 2 (Laratha) | 🟡 Masih memindahkan file (`shutil.move`) |
| `train.py` | Anggota 3 (Nur Aisya Jamil) | 🔴 **Gagal import model** (Bug B2) |
| `generate.py` | Anggota 3 (Nur Aisya Jamil) | 🔴 Gagal import model (B2) + salah panggil `decode` (Bug B3) |

**Kesimpulan:** setiap modul ditulis dengan baik secara terpisah, tetapi **alur end-to-end belum bisa dijalankan** karena 3 bug sambungan antar-modul (B1–B3). Ketiganya perbaikan kecil (1–3 baris).

---

## 🔴 Bug yang menghentikan alur (wajib diperbaiki dulu)

### B1 — `tokenizer.py` mengharapkan `build_vocab()` berupa dict
- **Lokasi:** `src/tokenizer.py`, baris ±24–30
- **Masalah:** `config.build_vocab()` mengembalikan **list** (indeks = ID, sesuai kontrak bagian 3), tetapi tokenizer memeriksa `isinstance(VOCAB, dict)` lalu melempar `TypeError`. Akibatnya **semua modul yang mengimpor tokenizer** (`dataset.py`, `train.py`, `generate.py`) ikut crash.
- **Bukti:**
  ```
  >>> type(config.build_vocab()).__name__
  'list'
  ```
- **Perbaikan (Anggota 2):** gunakan mapping yang sudah ada di config.
  ```python
  try:
      from .config import TOKEN_TO_ID
  except ImportError:
      from config import TOKEN_TO_ID
  # hapus blok `VOCAB = build_vocab()` dan pengecekan isinstance(dict)
  ```
- **Uji:** `python src/tokenizer.py data/raw/nottingham/ashover1.mid` harus mencetak `Self-test berhasil`.

### B2 — `train.py` & `generate.py` mengimpor kelas `MiniGPT` yang tidak ada
- **Lokasi:** `src/train.py` baris 26 & 31; `src/generate.py` baris 17 & 22 (ditandai `# SESUAIKAN nama kelas model`)
- **Masalah:** nama kelas di `model.py` (dan kontrak bagian 4) adalah `MiniMusicTransformer`.
- **Perbaikan (Anggota 3):** di keempat baris import,
  ```python
  from .model import MiniMusicTransformer as MiniGPT
  ```
  (atau ganti semua `MiniGPT` menjadi `MiniMusicTransformer`).

### B3 — `generate.py` memanggil `decode` dengan cara yang salah
- **Lokasi:** `src/generate.py`, fungsi `save_midi` (baris ±104–109)
- **Masalah:** kontrak: `decode(tokens, out_path) -> None` (langsung menulis file). Kode memanggil `tokenizer.decode(tokens)` tanpa `out_path` lalu memakai hasilnya `midi.write(...)` → `TypeError`.
- **Perbaikan (Anggota 3):**
  ```python
  def save_midi(ids: list[int], path: Path) -> None:
      special = {i for i in (PAD_ID, EOS_ID, BOS_ID) if i is not None}
      tokens = tokenizer.ids_to_tokens([i for i in ids if i not in special])
      tokenizer.decode(tokens, str(path))
  ```

---

## 🟡 Belum sesuai kontrak (perbaiki setelah alur jalan)

| No | Temuan | Pemilik | Usulan |
|---|---|---|---|
| K1 | `dataset.py` belum melakukan **augmentasi transposisi** (`TrainConfig.transpose_range` tidak dipakai) | Anggota 2 | Di `__getitem__` khusus train: geser semua ID `NOTE_*` acak −5..+5, jepit ke 0–127 (bisa memakai `tokenizer.transpose`) |
| K2 | `filter_midi_monophonic.py` masih `shutil.move` (memindahkan file) | Anggota 2 | Ganti menjadi hanya menulis laporan CSV. Data mentah sekarang sudah lengkap 1.034 file — jangan dijalankan lagi dengan `DRY_RUN = False` |
| K3 | Asumsi 2 (durasi 1–16) & 3 (panjang lagu) **belum dilaporkan** | Anggota 2 | `audit_midi.py` sudah menghitungnya — jalankan dan kirim ringkasan `duration_1_to_16_percent` & jumlah lagu > 256 token ke Ketua |
| K4 | Uji tokenizer baru menguji token ↔ ID, belum **round-trip musik** | Anggota 2 | Tambah cek `encode(decode(encode(f))) == encode(f)` |
| K5 | Anggota 3 push langsung ke `main` tanpa Pull Request | Anggota 3 | Perbaikan B2/B3 lewat branch + PR agar bisa direview |
| K6 | Belum ada log interaksi LLM dari Anggota 2 & 3 di `logs/` | Anggota 2, 3 | Setor log + share link chat ke Ketua/Anggota 4. Bug B1–B3 bagus dijadikan bagian **Debugging** |

---

## 🔵 Perbedaan dari kontrak yang **diusulkan diterima** (kontrak yang menyesuaikan)

Bagian ini butuh keputusan Ketua. Usulan: **diterima**, karena implementasinya konsisten satu sama lain dan masuk akal.

| No | Kontrak | Implementasi | Alasan diterima |
|---|---|---|---|
| D1 | `get_dataloaders(cfg)` | `create_dataloaders(config, model_config)` | Hanya beda nama; `train.py` sudah memakai nama baru |
| D2 | Batch berupa tuple `(x, y)`, label pad = `PAD_ID` | Batch berupa dict `{"input_ids", "labels", "attention_mask"}`, label pad = `-100` | `-100` adalah default `ignore_index` PyTorch; `train.py` konsisten memakai `IGNORE_INDEX = -100` |
| D3 | Tokenisasi sekali lewat `preprocessing.py` → `tokens.json` | Tokenisasi langsung saat `create_dataloaders` dipanggil | Dataset kecil (1.034 lagu), jadi masih cepat. `tokens.json` opsional |
| D4 | Dataset **tidak** di-commit | 1.034 file `.mid` di-commit ke `data/raw/nottingham/` (±4,4 MB) | Ukuran kecil dan memudahkan Colab: cukup `git clone`, data langsung tersedia |
| D5 | — | Folder `data/_excluded_non_monophonic/` berisi salinan 22 file | Diusulkan **dihapus dari repo** (duplikat; file aslinya sudah ada di `raw/` dan otomatis dilewati tokenizer) |

Jika disetujui, kontrak dinaikkan ke **v1.3** mengikuti tabel ini.

---

## ✅ Hal yang sudah bagus

- `model.py` dan `config.py` dipakai tanpa diubah oleh anggota lain.
- `tokenizer.py` memperhitungkan **perubahan tempo** saat mengubah detik ke not 1/16.
- `dataset.py` membagi train/val **per lagu sebelum dipotong** (tidak bocor), dengan seed.
- `train.py` lengkap: AdamW + warmup-cosine, grad clipping, label smoothing, early stopping, `metrics.csv`, grafik loss, checkpoint `best/last`, mode `--smoke`, dan `--run-name` untuk eksperimen tuning.
- `generate.py` mendukung temperature, top-k, greedy, dan prompt dari file MIDI.

---

## Urutan langkah berikutnya

1. Anggota 2 memperbaiki **B1**; Anggota 3 memperbaiki **B2 & B3** (lewat PR).
2. Ketua menjalankan **uji integrasi** di Colab:
   ```
   python src/tokenizer.py data/raw/nottingham/ashover1.mid
   python src/dataset.py
   python src/train.py --smoke --epochs 1
   python src/train.py --epochs 2 --run-name integrasi
   python src/generate.py --checkpoint outputs/checkpoints/best_integrasi.pt --n-samples 1
   ```
3. Setelah alur jalan: K1–K4, lalu training penuh & tuning oleh Anggota 3.
4. Ketua memperbarui kontrak ke v1.3 (bagian D).
