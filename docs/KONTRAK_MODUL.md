# 📜 Kontrak Antar-Modul — MIDI String Generator

Dokumen ini adalah **kesepakatan teknis** yang wajib diikuti semua anggota agar modul yang dibuat terpisah bisa langsung disambung.
Jika ingin mengubah sesuatu di sini, diskusikan dulu dengan Ketua lalu perbarui dokumen ini **dan** `src/config.py`.

Versi: 1.3 — 02/11/2026

| Versi | Perubahan |
|---|---|
| 1.0 | Kontrak awal |
| 1.1 | Tambah bagian 5 "Lokasi Data" |
| 1.2 | Tambah modul `src/preprocessing.py`, file `excluded_midi_report.csv`, format `stats.json`, dan status verifikasi asumsi 1 |
| 1.3 | Menyesuaikan kontrak dengan implementasi yang lulus uji integrasi #1: ketiga asumsi terverifikasi, keputusan D1–D5 dari `docs/REVIEW_INTEGRASI_01.md` (dataset di-commit, batch berbentuk dict, `create_dataloaders`, tokenisasi saat memuat data, folder `src/preprocessing/`) |

---

## 0. Keputusan Dasar

| Aspek | Keputusan | Catatan |
|---|---|---|
| Framework | **PyTorch** | Dijalankan di laptop atau Google Colab (GPU T4) |
| Dataset utama | **Nottingham**, folder `MIDI/melody` (1.034 lagu folk, melodi saja) | Di-commit ke repo di `data/raw/nottingham/` (±4,4 MB). Cadangan: **JSB Chorales** (Bach) |
| Library I/O MIDI | **`pretty_midi`** | Hanya untuk membaca & menulis file `.mid`. Tokenizer, vocab, dan augmentasi **ditulis sendiri** |
| Dilarang | Model pretrained, `nn.MultiheadAttention`/`nn.Transformer` bawaan, library tokenizer musik siap pakai (mis. `miditok`) | Agar memenuhi syarat "from scratch" |
| Arsitektur | Mini-Transformer **decoder-only** (gaya GPT / Music Transformer) | |
| Ukuran model | `d_model=256`, `n_head=4`, `n_layer=4`, `block_size=256`, `dropout=0.1` | Nilai resmi ada di `ModelConfig` (`src/config.py`) |
| Library lain | `numpy`, `matplotlib`, `tqdm` | Lihat `requirements.txt` |

Asumsi yang harus diverifikasi setelah dataset tersedia (tanggung jawab Anggota 2, hasilnya dilaporkan ke Ketua):

| No | Asumsi | Batas aman | Status |
|---|---|---|---|
| 1 | Setiap lagu memiliki track melodi tunggal (monofonik) | file non-monofonik < 5% | ✅ **Terverifikasi** 21/10/2026: 22 dari 1.034 file (2,1%) dikecualikan; 1.012 file dipakai (lihat `logs/04_keputusan_ketua.md`) |
| 2 | Sebagian besar durasi nada berada di rentang not 1/16 sampai not penuh (1–16) | nada terpotong (> 16) < 5% | ✅ **Terverifikasi** 02/11/2026: `DUR_16` hanya 0,15% (290 dari 193.940 token durasi) → `MAX_DURATION` tetap 16 |
| 3 | Panjang lagu dalam token cocok dengan `block_size = 256` | (informasi) | ✅ **Terverifikasi** 02/11/2026: median 328 token/lagu, 74,6% lagu > 256 token dan dipotong per jendela → `block_size` tetap 256 |

Semua asumsi terverifikasi (lihat `logs/04_keputusan_ketua.md` dan `logs/05_uji_integrasi.md`). Format token v1 dipakai tanpa perubahan.

---

## 1. Alur Data

```
data/raw/nottingham/*.mid
        │  dataset.create_dataloaders: split 90/10 per lagu
        ▼
tokenizer.encode ──▶ list[str] ──tokens_to_ids──▶ list[int]
        (file non-monofonik ditolak dengan ValueError → dilewati)
                                                           │
                                                 dataset.py (potong jendela T+1)
                                                           ▼
                     batch dict: input_ids, labels, attention_mask : LongTensor (B, T)
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
| `encode(midi_path: str) -> list[str]` | path file `.mid` | daftar token string (dengan `<BOS>`/`<EOS>`). Memilih track monofonik bernada dengan nada terbanyak; jika tidak ada → `ValueError` |
| `decode(tokens: list[str], out_path: str, tempo: float = 120.0) -> None` | daftar token | menulis file `.mid` (tidak mengembalikan objek) |
| `tokens_to_ids(tokens: list[str]) -> list[int]` | token string | ID integer |
| `ids_to_tokens(ids: list[int]) -> list[str]` | ID integer | token string |
| `transpose(tokens: list[str], semitones: int) -> list[str]` | token + geseran | token tergeser (augmentasi), pitch dijepit 0–127 |

- Vocabulary **wajib** diambil dari `config.TOKEN_TO_ID` (lihat Bug B1 di `docs/REVIEW_INTEGRASI_01.md`).
- Konversi detik → not 1/16 memperhitungkan perubahan tempo.
- Uji round-trip `encode(decode(encode(f))) == encode(f)`: **lulus 18/18** lagu monofonik (uji integrasi #1).
- Token yang tidak valid saat `decode` (misal dua `DUR` berturut-turut) dilewati, bukan menyebabkan crash.

### `src/preprocessing/` — Anggota 2
Skrip audit dataset yang dijalankan **manual**, bukan bagian dari alur training.

| File | Fungsi |
|---|---|
| `audit_midi.py` | Menghitung statistik per file & per track: monofonik/tidak, persentase durasi 1–16, estimasi jumlah token |
| `filter_midi_monophonic.py` | Mendata file non-monofonik ke laporan CSV |

- **Tidak boleh memindahkan, menghapus, atau mengubah** file di `data/raw/` (temuan K2: `shutil.move` harus diganti menjadi laporan saja).
- Penyaringan file non-monofonik saat training dilakukan **otomatis** oleh `tokenizer.encode` (melempar `ValueError`) dan `dataset.py` (melewati file tersebut). Hasil: 22 dari 1.034 file dilewati.
- `tokens.json` / `split.json` / `stats.json` **tidak wajib** (keputusan D3): dataset cukup kecil untuk ditokenisasi saat dimuat.

### `src/dataset.py` — Anggota 2
- Fungsi utama: `create_dataloaders(config: TrainConfig, model_config: ModelConfig) -> (train_loader, val_loader)` (keputusan D1).
- Membaca semua `*.mid` di bawah `config.data_dir` secara rekursif, **melewati** folder `_excluded_non_monophonic`.
- Pembagian **90/10 per lagu sebelum dipotong** (`config.val_ratio`, `config.seed`), agar lagu yang sama tidak muncul di train dan val.
- Kelas `MIDIStringDataset`: setiap lagu dipotong menjadi jendela `block_size + 1` token.
- Setiap item / batch berupa **dict** (keputusan D2):

| Kunci | Shape (batch) | Isi |
|---|---|---|
| `input_ids` | `(B, T)` long | token posisi `i … i+T-1`, pad kanan dengan `PAD_ID` |
| `labels` | `(B, T)` long | token posisi `i+1 … i+T` (digeser satu), pad kanan dengan **`-100`** |
| `attention_mask` | `(B, T)` long | 1 = token asli, 0 = padding (tidak dipakai model, karena causal + pad kanan) |

- **Augmentasi transposisi** acak `−config.transpose_range … +config.transpose_range` semiton, **khusus train** (temuan K1 — ⏳ belum diimplementasikan).
- Hasil uji integrasi #1: 931/103 lagu → 1.819/211 chunk; batch `(32, 256)`.

### `src/model.py` — Ketua
- Kelas `MiniMusicTransformer(nn.Module)`, dibuat dengan `MiniMusicTransformer(ModelConfig)`.
- `forward(idx: LongTensor (B, T)) -> logits: FloatTensor (B, T, V)`.
- `T` boleh ≤ `block_size`; jika lebih, lempar `ValueError`.
- Model **hanya mengembalikan logits**; perhitungan loss ada di `train.py`.
- Attention memakai **causal mask** (token tidak boleh melihat token setelahnya).

### `src/train.py` — Anggota 3
- Loss: `cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100, label_smoothing=...)`. Gunakan `.reshape`, **bukan** `.view` (lihat log 03, Eror 2.1).
- Optimizer AdamW (weight decay hanya untuk parameter ≥ 2D), warmup linear + cosine decay, gradient clipping, opsi early stopping.
- Dicatat per epoch: `train_loss`, `val_loss`, `val_perplexity = exp(val_loss)`, `val_accuracy` (tanpa padding), `lr`, `seconds`.
- Output (dengan `--run-name NAMA`, akhiran `_NAMA` ditambahkan):
  - `outputs/metrics_NAMA.csv`, `outputs/loss_curve_NAMA.png`
  - `outputs/checkpoints/best_NAMA.pt` (val_loss terendah) dan `last_NAMA.pt`
- Format checkpoint:
  ```python
  {
      "model_state": model.state_dict(),
      "model_config": asdict(model_cfg),
      "train_config": {...},   # TrainConfig + hyperparameter + run_name
      "epoch": epoch,
      "val_loss": val_loss,
  }
  ```
- Cara menjalankan: `python src/train.py --epochs 30 --run-name NAMA` (opsi lain: `--lr`, `--batch-size`, `--warmup-steps`, `--label-smoothing`, `--patience`, `--smoke`).
- File `*.pt` **tidak di-commit** (±17 MB per file, diabaikan `.gitignore`).

### `src/generate.py` — Anggota 3
- `generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> list[int]`.
- Jika sekuens lebih panjang dari `block_size`, hanya `block_size` token terakhir yang dimasukkan ke model.
- Berhenti saat menghasilkan `EOS_ID` atau mencapai `max_new_tokens`; `PAD` dan `BOS` tidak pernah dipilih; `temperature = 0` berarti greedy.
- Hasil disimpan ke `outputs/generated/*.mid` lewat `tokenizer.decode(tokens, out_path)` (lihat Bug B3).
- Cara menjalankan: `python src/generate.py --checkpoint outputs/checkpoints/best_NAMA.pt --n-samples 3 --temperature 0.95 --top-k 40` (opsional `--prompt-midi FILE` untuk melanjutkan lagu yang ada).

---

## 5. Lokasi Data

```
data/
├── raw/nottingham/              # 1.034 file .mid ASLI (di-commit, keputusan D4) — tidak boleh diubah
└── _excluded_non_monophonic/    # salinan 22 file non-monofonik — DIHAPUS dari repo (keputusan D5)
outputs/                         # dibuat otomatis oleh train.py & generate.py
├── metrics_<run>.csv
├── loss_curve_<run>.png
├── checkpoints/best_<run>.pt    # tidak di-commit
└── generated/*.mid
```

- Dataset **di-commit** ke repo (keputusan D4): ukurannya kecil (±4,4 MB) dan Colab cukup `git clone` untuk mendapatkan data.
- File non-monofonik **tidak perlu dipisahkan**; tokenizer menolak dan dataset melewatinya secara otomatis.
- Path relatif terhadap folder utama proyek di-resolve oleh `dataset.resolve_data_dir`, sehingga skrip bisa dijalankan dari mana saja di Windows maupun Colab.
- Properti path di `TrainConfig` (`raw_dir`, `processed_dir`, dll.) tetap tersedia untuk skrip tambahan.

---

## 6. Konvensi Kode

- Setiap fungsi punya docstring singkat dan komentar shape tensor, contoh `# (B, T, C)`.
- Seed acak diatur lewat `TrainConfig.seed` agar eksperimen bisa diulang.
- Path file memakai `pathlib.Path`, bukan string `"\\"` agar jalan di Windows & Colab.
- Pesan commit: `[nama] deskripsi singkat`.
