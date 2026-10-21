# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 21/10/2026
- **Anggota / Peran:** Ainun Widyah Astanti — Prompt Engineer & LLM Navigator (Ketua)
- **Target Modul:** `src/model.py` — Tahap 2 s.d. 7 (Multi-Head Attention sampai uji overfit)
- **Tautan Sesi Chat:** [tempel share link chat ini]

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:** "oke sudah set sisa model py, biar aku saja yang menguji namun berikan intruksi dan langkah langkahnya"
- **Alasan Khusus:**
  - Tahap 1 (attention satu head) sudah dipahami, sehingga sisa komponen diminta sekaligus agar pengerjaan efisien.
  - Pengujian sengaja dilakukan sendiri oleh Ketua di laptop/Colab, agar hasil eksekusi benar-benar dari lingkungan kelompok dan setiap error tercatat di log.
  - Setiap tahap tetap diberi tes terpisah (`python src/model.py --tahap N`) supaya jika ada bug, lokasinya langsung diketahui.
- **Output Model:** Lima komponen baru di `src/model.py` dan tujuh fungsi tes.

| Tahap | Komponen | Shape | Inti konsep |
|---|---|---|---|
| 2 | `MultiHeadAttention` | (B,T,C) → (B,T,C) | 4 head paralel, masing-masing 64 dimensi, digabung lalu `proj` |
| 3 | `FeedForward` | (B,T,C) → (B,T,4C) → (B,T,C) | MLP per token dengan GELU |
| 4 | `TransformerBlock` | (B,T,C) → (B,T,C) | Pre-LayerNorm + residual connection |
| 5 | `MiniMusicTransformer` | (B,T) → (B,T,V) | Token emb + positional emb → 4 blok → `lm_head` |
| 6 | Uji model lengkap | – | Jumlah parameter, causal, loss awal ≈ ln(V) |
| 7 | Uji overfit 1 batch | – | Model harus bisa menghafal 1 batch (loss → ~0) |

  Cuplikan krusial:
  ```python
  # Tahap 2 — gabungkan semua head
  out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)

  # Tahap 4 — pre-norm + residual
  x = x + self.attn(self.ln1(x))
  x = x + self.ff(self.ln2(x))

  # Tahap 5 — embedding token + posisi
  positions = torch.arange(T, device=idx.device)
  x = self.token_emb(idx) + self.pos_emb(positions)   # (B, T, C)
  ```

  Alasan desain penting:
  - **Pre-norm** (LayerNorm sebelum attention/FF) dipilih karena lebih stabil dilatih dibanding post-norm pada Transformer asli.
  - **Inisialisasi bobot std 0.02** (seperti GPT-2) agar loss awal mendekati tebakan acak `ln(148) ≈ 4.997`.
  - **Uji overfit 1 batch** adalah cara standar memastikan model *mampu* belajar sebelum training sungguhan. Jika satu batch kecil pun tidak bisa dihafal, pasti ada bug.

### Hasil Eksekusi Tes
Perintah: `python src/model.py --tahap N` dari folder utama proyek.

| Tahap | Lulus? | Catatan / angka penting |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | Jumlah parameter: ... ; loss awal: ... |
| 7 | ✅ (setelah perbaikan Eror 2.1) | Loss 5.0386 → 0.0018 dalam 300 step; akurasi hafalan 100%; device: cuda (Colab T4) |

Output lengkap:
```
[tempel output di sini]
```

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Eror 2.1 — `view` gagal pada tensor non-contiguous (Tahap 7)**
- **Lingkungan:** Google Colab, GPU T4 (`device: cuda`). Tahap 1–6 lulus, eror muncul di Tahap 7.
- **Pesan Eror dari Notebook:**
  ```
  === Tahap 7: Uji overfit 1 batch ===
  Model harus bisa 'menghafal' satu batch kecil. Jika gagal, ada bug di model.
  [7a] device: cuda | batch x (4, 32), y (4, 32)
  Traceback (most recent call last):
    File "/content/model.py", line 422, in <module>
      semua_tes[args.tahap - 1]()
      ~~~~~~~~~~~~~~~~~~~~~~~~~^^
    File "/content/model.py", line 398, in tes_tahap_7
      loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1), ignore_index=PAD_ID)
                                                              ~~~~~~^^^^
  RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
  ```
- **Prompt Perbaikan ke LLM:** Menempelkan seluruh traceback di atas tanpa tambahan penjelasan, agar LLM membaca baris dan variabel penyebabnya langsung dari pesan eror.
- **Analisis & Solusi Kelompok:**
  - **Penyebab:** `y = data[:, 1:]` adalah hasil *slicing* yang membuang kolom pertama. Tensor hasil slicing berbagi memori dengan `data`, sehingga elemen-elemennya tidak tersusun berurutan di memori (*non-contiguous*). Fungsi `.view()` hanya mengubah "cara membaca" memori tanpa menyalin data, sehingga gagal pada tensor non-contiguous.
  - **Kenapa `logits.view` tidak error?** `logits` adalah hasil baru dari `lm_head`, sehingga sudah contiguous. Masalahnya hanya pada `y`.
  - **Solusi:** mengganti `.view(-1)` menjadi `.reshape(-1)`, yang otomatis menyalin data bila tensor non-contiguous (alternatif: `y.contiguous().view(-1)`).
    ```python
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=PAD_ID)
    ```
  - **Dampak ke tim:** rumus loss di `docs/KONTRAK_MODUL.md` (bagian `train.py`) ikut diperbarui menjadi `.reshape`, agar Anggota 3 tidak mengalami eror yang sama.
  - **Pelajaran:** gunakan `.reshape` untuk tensor yang mungkin berasal dari slicing atau transpose; `.view` hanya aman untuk tensor contiguous.
- **Hasil setelah perbaikan:** Tahap 7 lulus. Loss turun dari 5.04 (setara tebakan acak, `ln(148) ≈ 4.997`) ke 0.0018 dengan akurasi hafalan 100%, membuktikan arsitektur benar dan model mampu belajar.
  ```
  === Tahap 7: Uji overfit 1 batch ===
  Model harus bisa 'menghafal' satu batch kecil. Jika gagal, ada bug di model.
  [7a] device: cuda | batch x (4, 32), y (4, 32)
  [7b] step    1 | loss 5.0386
  [7b] step   50 | loss 0.0160
  [7b] step  100 | loss 0.0068
  [7b] step  150 | loss 0.0044
  [7b] step  200 | loss 0.0031
  [7b] step  250 | loss 0.0023
  [7b] step  300 | loss 0.0018
  [7c] loss akhir: 0.0018 | akurasi hafalan: 100.0%
  Tahap 7 LULUS ✅
  ```
