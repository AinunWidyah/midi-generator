# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 02/11/2026
- **Anggota / Peran:** Ainun Widyah Astanti — Ketua (Prompt Engineer & LLM Navigator) merangkap Anggota 4 (Documentarian & Error Handler)
- **Target Modul:** Review integrasi seluruh modul + uji end-to-end (`tokenizer.py` → `dataset.py` → `model.py` → `train.py` → `generate.py`)
- **Tautan Sesi Chat:** [https://claude.ai/share/6d62c99b-b497-4275-9a3e-38719a5947e8]
- **Bukti eksekusi:** `notebooks/uji_integrasi_01.ipynb` (Google Colab, Tesla T4, PyTorch 2.11.0+cu130)

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI — Review kode seluruh anggota

- **Prompt Kelompok:** "oke next semua anggota kelompok sudah ngirim code nya di github apa yang harus dilakukan sekarang", lalu mengizinkan LLM membaca repository GitHub kelompok.
- **Alasan Khusus:** Setiap anggota menulis modulnya secara terpisah. Sebelum training penuh, Ketua perlu memastikan modul-modul tersebut **tersambung** sesuai `docs/KONTRAK_MODUL.md`, bukan hanya berjalan sendiri-sendiri.
- **Output Model:** `docs/REVIEW_INTEGRASI_01.md` berisi:
  - **3 bug sambungan (B1–B3)** yang membuat alur tidak bisa dijalankan sama sekali;
  - 6 temuan belum sesuai kontrak (K1–K6);
  - 5 perbedaan dari kontrak yang diusulkan diterima (D1–D5).

### 2. TAHAP PENANGANAN EROR / DEBUGGING — Bug sambungan antar-modul

**Bug B1 — `tokenizer.py` crash saat di-import (pemilik: Anggota 2)**
- **Temuan:** `tokenizer.py` mengharuskan `build_vocab()` berupa dict:
  ```python
  VOCAB = build_vocab()
  if not isinstance(VOCAB, dict):
      raise TypeError("build_vocab() harus mengembalikan dict dengan format {token: id}.")
  ```
  padahal sesuai kontrak bagian 3, `build_vocab()` mengembalikan **list** (indeks = ID). Bukti:
  ```
  >>> type(config.build_vocab()).__name__
  'list'
  ```
  Karena `dataset.py`, `train.py`, dan `generate.py` semuanya mengimpor tokenizer, seluruh alur ikut gagal.
- **Analisis:** Anggota 2 kemungkinan menguji tokenizer dengan versi `config.py` yang berbeda atau berasumsi format vocab tanpa membaca kontrak. Ini contoh klasik **error integrasi**: tiap modul benar menurut asumsinya sendiri.
- **Solusi:** memakai mapping yang sudah ada di config — `from config import TOKEN_TO_ID` — dan menghapus pengecekan dict. Diperbaiki Anggota 2 pada commit `5f48847` (02/11 14:04).

**Bug B2 — `train.py` & `generate.py` mengimpor kelas `MiniGPT` yang tidak ada (pemilik: Anggota 3)**
- **Temuan:** `from .model import MiniGPT  # SESUAIKAN nama kelas model`, sedangkan nama kelas di `model.py` dan kontrak adalah `MiniMusicTransformer`. Komentar `SESUAIKAN` menunjukkan kode dari LLM yang belum disesuaikan dengan proyek.
- **Solusi:** semua import dan pemanggilan diganti menjadi `MiniMusicTransformer`. Diperbaiki Anggota 3 pada commit `c0bc3ad` & `eb7dc9b` (02/11 13:36).

**Bug B3 — `generate.py` memanggil `decode` dengan cara yang salah (pemilik: Anggota 3)**
- **Temuan:** `midi = tokenizer.decode(tokens)` lalu `midi.write(...)`, padahal kontrak: `decode(tokens, out_path) -> None` (langsung menulis file).
- **Solusi:** `tokenizer.decode(tokens, str(path))`. Diperbaiki pada commit `eb7dc9b`.

**Pelajaran kelompok:**
1. Kontrak antar-modul hanya berguna jika **benar-benar dibaca** sebelum vibe coding; ketiga bug terjadi tepat di titik sambungan yang diatur kontrak.
2. Kode dari LLM yang mengandung komentar `SESUAIKAN`/`TODO` wajib diperiksa sebelum di-push.
3. Perbaikan B1–B3 di-push langsung ke `main` tanpa Pull Request; ke depan semua perubahan lewat PR agar bisa direview.

### 3. UJI INTEGRASI END-TO-END (setelah B1–B3 diperbaiki)

- **Prompt Kelompok:** "berarti tugas aku ya sebagai ketua" → LLM menyusun notebook `notebooks/uji_integrasi_01.ipynb` (6 tahap), dijalankan Ketua di Colab.
- **Commit yang diuji:** `5f48847` (Laratha, 02/11 14:04)

| Tahap | Modul | Lulus? | Angka / catatan penting |
|---|---|---|---|
| 0 | Persiapan | ✅ | Tesla T4, PyTorch 2.11.0+cu130, pretty_midi 0.2.11; 1.034 file MIDI |
| 1 | `tokenizer.py` | ✅ | Self-test lulus (ashover1: 140 token). Round-trip encode→decode→encode identik **18/18** lagu monofonik; 2 file (ashover15, ashover19) ditolak karena non-monofonik, sesuai daftar karantina |
| 2 | Asumsi 2 & 3 | ✅ | `DUR_16` **0,15%** (290/193.940); token per lagu min 36 / median 328 / rata-rata 385 / maks 5.452; 74,6% lagu > 256 token (dipotong per jendela) |
| 3 | `model.py` | ✅ | **3.297.792 parameter**; loss awal 5,039 (teori ln 148 = 4,997); causal terjaga |
| 4 | `dataset.py` + model | ✅ | 931/103 lagu train/val; **1.819 / 211 chunk**; batch (32, 256) int64 → logits (32, 256, 148); label = input digeser satu; 22 file dilewati |
| 5 | `train.py` | ✅ | Smoke: loss 5,05, acc 0,8% (≈ acak). Data asli epoch 2: train 2,076, **val 1,716**, ppl 5,56, **acc 45,3%**; ±5 detik/epoch |
| 6 | `generate.py` | ✅ | 2 sampel × 512 token (T=0,95, top-k=40): 246 nada/65,6 detik dan 244 nada/124,6 detik; dapat diputar |

Metrik training (data asli):
```
epoch  train_loss  val_loss  val_perplexity  val_accuracy       lr  seconds
    1     3.54023   2.44977         11.5857       0.42483 0.000087      5.1
    2     2.07620   1.71593          5.5618       0.45333 0.000172      4.7
```

### 4. KEPUTUSAN KETUA

1. **Ketiga asumsi kontrak terverifikasi:**
   - Asumsi 1 (monofonik): 22 file non-monofonik — angka **sama persis** di audit Anggota 2, Tahap 2, dan Tahap 4.
   - Asumsi 2 (durasi): `DUR_16` hanya 0,15% → **`MAX_DURATION` tetap 16**.
   - Asumsi 3 (panjang): median 328 token, lagu panjang dipotong per jendela → **`block_size` tetap 256**.
   - Kesimpulan: `config.py` dan `model.py` **tidak diubah**; format token v1 cocok untuk Nottingham.
2. **Alur end-to-end dinyatakan berjalan** → Anggota 3 boleh mulai training penuh & tuning.
3. **Augmentasi transposisi (K1) diprioritaskan** sebelum training penuh, karena data kecil (±57 langkah/epoch untuk model 3,3 juta parameter) sehingga rawan overfitting.

### 5. CATATAN UNTUK EVALUASI & LAPORAN

- **Akurasi 45% perlu dibaca relatif terhadap baseline:** `DUR_2` (not 1/8) mencakup ±65% token durasi (khas jig & reel). Model yang selalu menebak `DUR_2` di posisi durasi sudah memperoleh ±32% akurasi total, sehingga 45% berarti model baru sedikit melampaui pola paling umum.
- **val_loss < train_loss di epoch awal itu wajar:** train_loss dirata-rata selama epoch saat model masih berkembang, dan dropout aktif saat training.
- **Learning rate belum mencapai nilai penuh** (0,000172 dari 0,0003) karena warmup 200 langkah lebih panjang dari 2 epoch (±114 langkah). Tidak berpengaruh pada training penuh.
- **Durasi sampel 2 hampir dua kali sampel 1** dengan jumlah nada sama → model epoch 2 masih sering menghasilkan jeda/durasi panjang. Dipakai sebagai pembanding "awal training" saat presentasi.
- **Outlier:** satu lagu sepanjang 5.452 token (±14× median).
- **Peringatan berulang** `RuntimeWarning: Gagal memproses ... non-monofonik` (22 baris) saat membuat dataset — tidak berbahaya, tetapi bisa dirapikan (temuan K2).

---

### 6. UJI ULANG SETELAH AUGMENTASI TRANSPOSISI (K1) & BASELINE

- **Commit yang diuji:** `14957e9` (merge PR #6, 02/11 15:31), mencakup PR #4 & #5 (augmentasi transposisi, Laratha) dan PR #6 (`baseline.py`, Nur Aisya).
- **Catatan proses:** PR #4 berisi `SyntaxError` (`chunk = tokenizer.tokens_to_ids(tokens)chunk = self.examples[index]`) dan sempat di-merge oleh pembuatnya sendiri sebelum direview, sehingga `main` rusak ±20 menit (14:53–15:12) sampai PR #5 memperbaikinya. Tindak lanjut: PR hanya di-merge Ketua setelah review; disarankan mengaktifkan proteksi branch `main`.

| Cek | Sebelum K1 | Sesudah K1 | Keterangan |
|---|---|---|---|
| Rentang ID batch train | 0–147 | **0–147** ✅ | Transposisi tidak menghasilkan token di luar vocab |
| val_loss epoch 2 | 1,716 | 1,806 | Sedikit naik: data train kini bervariasi (transposisi acak), tugas jadi lebih sulit dalam jangka pendek. Manfaat augmentasi (mengurangi overfitting) baru terlihat pada training panjang |
| val_accuracy epoch 2 | 45,3% | 44,5% | idem |
| Generate | ✅ | ✅ | 2 sampel × 512 token |

**Baseline tanpa belajar (`src/baseline.py`, data validasi, 40.536 token target):**

| Baseline | Akurasi |
|---|---|
| A — selalu token paling umum (`DUR_2`) | 33,4% |
| **B — paling umum menurut tipe token sebelumnya** | **40,7%** |
| C — bigram (paling umum setelah token sebelumnya) | 37,5% |
| Model, epoch 2 | 44,5% |

- Token durasi = 49,9% dari target; `DUR_2` = 67,0% dari token durasi (konsisten dengan Tahap 2).
- **Analisis:** setelah 2 epoch, model baru ±4 poin di atas baseline terbaik (B). Training penuh dinilai berhasil jika `val_accuracy` **jelas melampaui 40,7%**; target ini dipakai di laporan.
- **Catatan metodologi:** statistik baseline diambil dari `train_loader` yang kini memakai augmentasi acak, sehingga bigram (C) menjadi "kabur" dan justru lebih rendah dari B. Disarankan baseline memakai data train **tanpa** augmentasi agar adil (`MIDIStringDataset(..., augment=False)`).

**Keputusan Ketua:** alur tetap berjalan setelah K1 → Anggota 3 dipersilakan memulai training penuh (30 epoch) dan eksperimen tuning.
