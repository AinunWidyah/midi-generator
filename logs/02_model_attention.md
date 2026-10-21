# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 21/10/2026
- **Anggota / Peran:** Ainun Widyah Astanti — Prompt Engineer & LLM Navigator (Ketua)
- **Target Modul:** `src/model.py` — Tahap 1: Causal Self-Attention (satu head)
- **Tautan Sesi Chat:** [tempel share link chat ini]

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:** "boleh dan aku juga mulai jangan lupakan [template Log Interaksi LLM] karna aku sudah mulai" — sebagai jawaban atas tawaran LLM untuk membangun `model.py` bertahap, dimulai dari Causal Self-Attention.
- **Alasan Khusus:**
  - Model dibangun **per komponen**, bukan sekaligus, agar setiap bagian bisa diuji dan dipahami sebelum digabung. Ini juga memudahkan debugging karena kesalahan langsung terlokalisasi.
  - Attention dikerjakan pertama karena merupakan inti Transformer dan syarat "dari nol" pada tugas (tidak memakai `nn.MultiheadAttention` bawaan PyTorch).
  - Satu head dulu, baru dikembangkan menjadi multi-head di Tahap 2, agar alur Q, K, V mudah diikuti.
- **Output Model:** Kelas `CausalSelfAttentionHead` di `src/model.py` beserta tiga tes otomatis.

  Cuplikan krusial:
  ```python
  scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, T, T)
  mask = self.causal_mask[:T, :T]
  scores = scores.masked_fill(mask == 0, float("-inf"))
  weights = F.softmax(scores, dim=-1)
  out = weights @ v  # (B, T, hs)
  ```

  Penjelasan konsep:
  - **Q, K, V**: setiap token diproyeksikan menjadi Query (apa yang dicari), Key (apa yang dimiliki), dan Value (informasi yang diberikan).
  - **Pembagian `sqrt(head_size)`**: mencegah skor terlalu besar sehingga softmax tidak "jenuh" dan gradien tetap sehat.
  - **Causal mask (`torch.tril`)**: nada ke-i hanya boleh melihat nada 0..i. Tanpa mask, model "menyontek" nada berikutnya saat training dan tidak berguna saat generate.
  - **`register_buffer`**: mask ikut pindah ke GPU dan tersimpan di checkpoint, tetapi tidak ikut dilatih.

  Tes yang disertakan (`python src/model.py`):
  1. Shape: input `(2, 8, 256)` → output `(2, 8, 64)`.
  2. Causal: mengubah token terakhir tidak mengubah output token sebelumnya.
  3. Visualisasi bobot attention: matriks harus segitiga bawah.

- **Hasil Eksekusi Tes:** [tempel output `python src/model.py` di sini]

### 2. TAHAP PENANGANAN EROR / DEBUGGING
- **Pesan Eror dari Notebook:** [isi jika ada eror saat menjalankan tes]
- **Prompt Perbaikan ke LLM:** "..."
- **Analisis & Solusi Kelompok:** ...
