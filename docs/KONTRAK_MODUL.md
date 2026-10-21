# 📜 Kontrak Antar-Modul — MIDI String Generator

Dokumen ini adalah **kesepakatan teknis** yang wajib diikuti semua anggota agar modul yang dibuat terpisah bisa langsung disambung.
Jika ingin mengubah sesuatu di sini, diskusikan dulu dengan Ketua lalu perbarui dokumen ini **dan** `src/config.py`.

Versi: 1.0 — 21/10/2026

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
1. Setiap lagu memiliki track melodi tunggal (monofonik).
2. Sebagian besar durasi nada berada di rentang not 1/16 sampai not penuh (1–16).
3. Panjang lagu dalam token cocok dengan `block_size = 256`.

Jika ada asumsi yang meleset, kontrak dinaikkan ke versi 1.1 dan perubahannya dicatat di log.

---

## 1. Alur Data

```
.mid ──tokenizer.encode──▶ list[str] ──tokens_to_ids──▶ list[int]
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
- Loss: `cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=PAD_ID)`, dengan opsi `label_smoothing`.
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

## 5. Konvensi Kode

- Setiap fungsi punya docstring singkat dan komentar shape tensor, contoh `# (B, T, C)`.
- Seed acak diatur lewat `TrainConfig.seed` agar eksperimen bisa diulang.
- Path file memakai `pathlib.Path`, bukan string `"\\"` agar jalan di Windows & Colab.
- Pesan commit: `[nama] deskripsi singkat`.
