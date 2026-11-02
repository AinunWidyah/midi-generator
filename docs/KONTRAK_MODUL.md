# 📜 Kontrak Antar-Modul — MIDI String Generator

Dokumen ini adalah **kesepakatan teknis** yang wajib diikuti semua anggota agar modul yang dibuat terpisah bisa langsung disambung.
Jika ingin mengubah sesuatu di sini, diskusikan dulu dengan Ketua lalu perbarui dokumen ini **dan** `src/config.py`.

Versi: 1.2 — 21/10/2026

| Versi | Perubahan |
|---|---|
| 1.0 | Kontrak awal |
| 1.1 | Tambah bagian 5 "Lokasi Data" |
| 1.2 | Tambah modul `src/preprocessing.py`, file `excluded_midi_report.csv`, format `stats.json`, dan status verifikasi asumsi 1 |

---

## 0. Keputusan Dasar

| Aspek | Keputusan | Catatan |
|---|---|---|
| Framework | **PyTorch** | Dijalankan di laptop atau Google Colab (GPU T4) |
| Dataset utama | **Nottingham** (lagu folk, melodi + akor) | Cadangan: **JSB Chorales** (Bach). Asumsi format token di bawah wajib dicek Anggota 2 setelah dataset diunduh |
| Library I/O MIDI | **`pretty_midi`** | Hanya untuk membaca & menulis file `.mid`. Tokenizer, vocab, dan augmentasi **ditulis sendiri** |
| Dilarang | Model pretrained, `nn.MultiheadAttention`/`nn.Transformer` bawaan, library tokenizer musik siap pakai (mis. `miditok`) | Agar memenuhi syarat "from scratch" |
| Arsitektur | Mini-Transformer **decoder-only** (gaya GPT / Music Transformer) | |
| Ukuran model | `d_model=256`, `n_head=4`, `n_layer=4`, `block_size=256`, `dropout=0.1` | Nilai resmi ada di `ModelConfig` (`src/config.py`) |
| Library lain | `numpy`, `matplotlib`, `tqdm` | Lihat `requirements.txt` |

Asumsi yang harus diverifikasi setelah dataset tersedia (tanggung jawab Anggota 2, hasilnya dilaporkan ke Ketua):

| No | Asumsi | Batas aman | Status |
|---|---|---|---|
| 1 | Setiap lagu memiliki track melodi tunggal (monofonik) | file non-monofonik < 5% | ✅ **Terverifikasi** 21/10/2026: 22 dari 1.034 file (2,1%) dikecualikan; 1.012 file dipakai (lihat `logs/04_keputusan_ketua.md`) |
| 2 | Sebagian besar durasi nada berada di rentang not 1/16 sampai not penuh (1–16) | nada terpotong (> 16) < 5% | ⏳ Menunggu `tokenizer.py` |
| 3 | Panjang lagu dalam token cocok dengan `block_size = 256` | (informasi) | ⏳ Menunggu `tokenizer.py` |

Jika ada asumsi yang meleset, kontrak dinaikkan ke versi berikutnya dan perubahannya dicatat di log.

---

## 1. Alur Data

```
data/raw/*.mid ──preprocessing.py (filter + tokenizer.encode)──▶ data/processed/tokens.json
                                                           │
                                     tokens_to_ids  ◀──────┘
                                           ▼
                                       list[int]
                                                           │
                                                 dataset.py (potong jendela)
                                                           ▼
                                        x, y : LongTensor (B, T)
                                                           │
                                                      model.py
                                                           ▼
                                        logits : FloatTensor (B, T, V)
                                                           │
                                     train.py (loss)   generate.py (sampling)
                                                           ▼
                        list[int] ──ids_to_tokens──▶ list[str] ──tokenizer.decode──▶ .mid
```

Keterangan simbol: `B` = batch size, `T` = panjang sekuens (`block_size`), `V` = ukuran vocabulary.

---

## 2. Format Token (v1: melodi monofonik)

Versi pertama hanya memakai **melodi** (satu nada pada satu waktu). Akor/polifoni adalah pengembangan lanjutan.

Setiap peristiwa musik ditulis sebagai **pasangan token**:

| Token | Arti | Contoh |
|---|---|---|
| `NOTE_{p}` | Nada dengan nomor pitch MIDI `p` (0–127) | `NOTE_60` = C tengah |
| `REST` | Jeda / diam | `REST` |
| `DUR_{d}` | Durasi peristiwa sebelumnya, dalam satuan **not 1/16** (1–16) | `DUR_4` = 1 ketuk (not 1/4) |

Contoh satu birama 4/4:

```
<BOS> NOTE_67 DUR_4 NOTE_69 DUR_2 NOTE_71 DUR_2 REST DUR_4 NOTE_72 DUR_4 ... <EOS>
```

Aturan:
1. Token selalu berpola **`NOTE/REST` lalu `DUR`**, bergantian.
2. Durasi dibulatkan (kuantisasi) ke not 1/16 terdekat. Durasi < 1 menjadi 1, durasi > 16 dipotong menjadi 16.
3. Jeda antar-nada yang lebih dari 0 dicatat sebagai `REST DUR_{d}`.
4. Setiap lagu diawali `<BOS>` dan diakhiri `<EOS>`.

---

## 3. Vocabulary & Token Khusus

| ID | Token | Fungsi |
|---|---|---|
| 0 | `<PAD>` | Pengisi agar panjang sekuens sama; **diabaikan oleh loss** |
| 1 | `<BOS>` | Awal lagu |
| 2 | `<EOS>` | Akhir lagu |
| 3–130 | `NOTE_0` … `NOTE_127` | 128 pitch MIDI |
| 131 | `REST` | Jeda |
| 132–147 | `DUR_1` … `DUR_16` | 16 nilai durasi |

**Ukuran vocab: V = 148** (tetap, tidak bergantung dataset).
Semua 128 pitch disertakan agar augmentasi transposisi tidak pernah menghasilkan token di luar vocab.
Urutan ID dibuat oleh `build_vocab()` di `src/config.py` dan **tidak boleh diubah** setelah training dimulai.

---

## 4. Kontrak per Modul

### `src/config.py` — Ketua
- Berisi `ModelConfig`, `TrainConfig`, konstanta token khusus, dan `build_vocab()`.
- Semua modul mengimpor nilai dari sini, **tidak boleh** menulis ulang angka secara manual.

### `src/tokenizer.py` — Anggota 2
| Fungsi | Input | Output |
|---|---|---|
| `encode(midi_path: str) -> list[str]` | path file `.mid` | daftar token string (dengan `<BOS>`/`<EOS>`) |
| `decode(tokens: list[str], out_path: str, tempo: float = 120.0) -> None` | daftar token | menulis file `.mid` |
| `tokens_to_ids(tokens: list[str]) -> list[int]` | token string | ID integer |
| `ids_to_tokens(ids: list[int]) -> list[str]` | ID integer | token string |
| `transpose(tokens: list[str], semitones: int) -> list[str]` | token + geseran | token tergeser (augmentasi) |

Wajib ada **uji round-trip**: `decode(encode(f))` harus terdengar sama dengan `f`.
Token yang tidak valid saat `decode` (misal dua `DUR` berturut-turut) dilewati, bukan menyebabkan crash.

### `src/preprocessing.py` — Anggota 2
Skrip yang dijalankan **sekali** (dan bisa diulang kapan saja) untuk mengubah data mentah menjadi data siap pakai.

- Cara menjalankan (dari folder utama): `python src/preprocessing.py` (opsional `--dataset jsb_chorales`).
- Input: semua `*.mid` di `cfg.raw_dir`, **diurutkan berdasarkan nama** agar hasil selalu sama.
- **Tidak boleh memindahkan, menghapus, atau mengubah** file di `raw/`. File yang dikecualikan cukup dicatat.
- Langkah wajib, berurutan:
  1. **Filter monofonik**: logika dari `filter_midi_monophonic.py` dipindahkan ke sini. File yang tidak lolos dicatat ke `cfg.excluded_path` beserta alasannya. Kriteria persis "tidak monofonik" ditulis di docstring.
  2. **Tokenisasi**: `tokenizer.encode` untuk setiap file yang lolos, lalu simpan ke `cfg.tokens_path`.
  3. **Split 90/10 per lagu**: acak memakai `random.Random(cfg.seed)` dengan rasio `cfg.val_ratio`, lalu simpan ke `cfg.split_path`.
  4. **Statistik & cek asumsi 2–3**: simpan ke `cfg.stats_path` dengan format di bawah.
  5. Cetak ringkasan ke layar (jumlah file, dikecualikan, gagal, train/val, dan hasil cek asumsi).
- File yang gagal dibaca **tidak menghentikan** skrip: dicatat di `excluded_midi_report.csv` dengan alasan `gagal_dibaca`.

Format `stats.json` (kunci wajib):
```json
{
  "dataset": "nottingham",
  "total_files": 1034,
  "excluded": {"non_monophonic": 22, "gagal_dibaca": 0},
  "kept": 1012,
  "pct_non_monophonic": 2.13,
  "total_notes": 0,
  "pct_duration_clipped": 0.0,
  "duration_histogram": {"1": 0, "2": 0, "...": 0, "16": 0, ">16": 0},
  "tokens_per_song": {"min": 0, "mean": 0.0, "median": 0, "max": 0},
  "pitch_range": {"min": 0, "max": 0},
  "split": {"train": 0, "val": 0}
}
```
(Angka 0 di atas hanya contoh format; nilai asli diisi oleh skrip.)

Format `excluded_midi_report.csv`: kolom `file,alasan,detail`, contoh `abc_12.mid,non_monophonic,overlap pada 14 nada`.

### `src/dataset.py` — Anggota 2
- Kelas `MidiDataset(torch.utils.data.Dataset)`.
- `__getitem__` mengembalikan tuple `(x, y)`:
  - `x`: `LongTensor` shape `(T,)` → token posisi `i … i+T-1`
  - `y`: `LongTensor` shape `(T,)` → token posisi `i+1 … i+T` (digeser satu)
- Sekuens yang lebih pendek dari `T+1` di-pad dengan `PAD_ID` di kanan.
- Fungsi `get_dataloaders(cfg) -> (train_loader, val_loader)` dengan pembagian **90/10 per lagu** (bukan per potongan, agar tidak bocor).
- Output DataLoader: `x, y` shape `(B, T)`, dtype `torch.long`.

### `src/model.py` — Ketua
- Kelas `MiniMusicTransformer(nn.Module)`, dibuat dengan `MiniMusicTransformer(ModelConfig)`.
- `forward(idx: LongTensor (B, T)) -> logits: FloatTensor (B, T, V)`.
- `T` boleh ≤ `block_size`; jika lebih, lempar `ValueError`.
- Model **hanya mengembalikan logits**; perhitungan loss ada di `train.py`.
- Attention memakai **causal mask** (token tidak boleh melihat token setelahnya).

### `src/train.py` — Anggota 3
- Loss: `cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD_ID)`, dengan opsi `label_smoothing`. Gunakan `.reshape`, **bukan** `.view`, karena `y` hasil slicing (`data[:, 1:]`) tidak contiguous di memori (lihat log 03, Eror 2.1).
- Dicatat per epoch: `train_loss`, `val_loss`, `val_perplexity = exp(val_loss)`, `val_accuracy` (tanpa PAD).
- Riwayat metrik disimpan ke `outputs/metrics.csv`; grafik ke `outputs/loss_curve.png`.
- Format checkpoint (`outputs/checkpoints/*.pt`):
  ```python
  {
      "model_state": model.state_dict(),
      "model_config": asdict(model_cfg),
      "train_config": asdict(train_cfg),
      "epoch": epoch,
      "val_loss": val_loss,
  }
  ```

### `src/generate.py` — Anggota 3
- `generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> list[int]`.
- Jika sekuens lebih panjang dari `block_size`, hanya `block_size` token terakhir yang dimasukkan ke model.
- Berhenti saat menghasilkan `EOS_ID` atau mencapai `max_new_tokens`.
- Hasil disimpan ke `outputs/generated/*.mid` lewat `tokenizer.decode`.

---

## 5. Lokasi Data

```
data/
├── README.md                     # sumber dataset & cara mengunduh (satu-satunya file data yang di-commit)
├── raw/<dataset_name>/           # file .mid ASLI — tidak boleh diubah
└── processed/<dataset_name>/     # hasil preprocessing — boleh dihapus & dibuat ulang
    ├── tokens.json               # {"nama_lagu.mid": ["<BOS>", "NOTE_67", "DUR_4", ..., "<EOS>"], ...}
    ├── split.json                # {"train": ["a.mid", ...], "val": ["b.mid", ...]}
    ├── stats.json                # hasil cek asumsi (format di bagian 4, preprocessing.py)
    └── excluded_midi_report.csv  # daftar file yang dikecualikan + alasannya
```

- `<dataset_name>` mengikuti `TrainConfig.dataset_name` (`nottingham` atau `jsb_chorales`).
- Path **tidak boleh ditulis manual**; gunakan properti di `TrainConfig`: `raw_dir`, `processed_dir`, `tokens_path`, `split_path`, `stats_path`, `excluded_path`.
- Semua isi `processed/` dibuat oleh `python src/preprocessing.py`; tidak ada file yang dibuat atau diedit dengan tangan.
- Isi `data/` (kecuali `README.md` dan `.gitkeep`) **tidak di-commit** karena diabaikan `.gitignore`. Anggota lain mengambil dataset dari sumber di `data/README.md` atau dari folder Google Drive kelompok.
- `tokens.json` disimpan sebagai token **string** (bukan ID) agar mudah dibaca & diperiksa manual; konversi ke ID dilakukan di `dataset.py` lewat `tokens_to_ids`.
- Augmentasi transposisi dilakukan **saat training** di `dataset.py`, tidak disimpan ke `processed/`.

---

## 6. Konvensi Kode

- Setiap fungsi punya docstring singkat dan komentar shape tensor, contoh `# (B, T, C)`.
- Seed acak diatur lewat `TrainConfig.seed` agar eksperimen bisa diulang.
- Path file memakai `pathlib.Path`, bukan string `"\\"` agar jalan di Windows & Colab.
- Pesan commit: `[nama] deskripsi singkat`.
