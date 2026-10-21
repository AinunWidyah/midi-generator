# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 21/10/2026
- **Anggota / Peran:** Ainun Widyah Astanti — Prompt Engineer & LLM Navigator (Ketua)
- **Target Modul:** Perencanaan proyek, pembagian jobdesk, setup repository GitHub, dan kontrak antar-modul
- **Tautan Sesi Chat:** [tempel share link chat ini]

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)

**Prompt 1.1**
- **Prompt Kelompok:** "bantu aku mengerjakan tugas ini langkah langkah awal dan jobdesc yang harus dilakukan, aku memilih project Generative Audio & Sequence • Generative Audio (MIDI String Generator)"
- **Alasan Khusus:** Memahami gambaran besar proyek dan membagi tugas sebelum mulai menulis kode, agar setiap anggota tahu tanggung jawabnya sejak awal.
- **Output Model:** Rekomendasi arsitektur Mini-Transformer decoder-only (bukan LSTM, agar memenuhi syarat "model terkini 10 tahun terakhir"), dataset ringan (Nottingham / JSB Chorales), alur MIDI → token → model → MIDI, jobdesk detail per anggota, dan contoh system prompt.

**Prompt 1.2**
- **Prompt Kelompok:** "aku sebagai 1 – Prompt Engineer & LLM Navigator (Ketua), ingin terlebih dahulu membuat github"
- **Alasan Khusus:** Repository perlu ada lebih dulu agar kontribusi tiap anggota tercatat lewat Git sejak awal (kriteria Kerjasama Tim 20%).
- **Output Model:** Kerangka repository dalam bentuk zip (README, .gitignore, requirements.txt, struktur folder `data/ src/ notebooks/ outputs/ logs/`, template log) serta langkah `git init` → `commit` → `push`.

**Prompt 1.3**
- **Prompt Kelompok:** "setelah github aku sebagai ketua atau prompt engineer & LLM navigator harus ngapain ya terus juga kan model.py juga kan aku"
- **Alasan Khusus:** Menentukan prioritas Ketua setelah repo siap, karena anggota lain tidak bisa mulai sebelum ada kesepakatan teknis.
- **Output Model:** Dokumen `docs/KONTRAK_MODUL.md` dan `src/config.py`, berisi:
  - Format token berpasangan `NOTE_{pitch} DUR_{durasi}` (melodi monofonik, durasi satuan not 1/16).
  - Vocabulary tetap 148 token (PAD=0, BOS=1, EOS=2, 128 NOTE, REST, 16 DUR).
  - Shape tensor antar-modul: `x, y: (B, T)` → `logits: (B, T, V)`.
  - Rencana pembangunan `model.py` dalam 7 tahap.

  Cuplikan krusial:
  ```python
  def build_vocab() -> list[str]:
      vocab = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
      vocab += [f"NOTE_{p}" for p in range(NUM_PITCHES)]
      vocab += [REST_TOKEN]
      vocab += [f"DUR_{d}" for d in range(1, MAX_DURATION + 1)]
      return vocab
  ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Eror 2.1 — Struktur folder ganda setelah ekstrak zip**
- **Pesan Eror dari Konsol:** (bukan eror, tetapi output commit menunjukkan path yang salah)
  ```
  [master (root-commit) e752bd3] [ketua] setup struktur awal proyek
   14 files changed, 130 insertions(+)
   create mode 100644 midi-generator/.gitignore
   create mode 100644 midi-generator/README.md
   ...
  ```
- **Prompt Perbaikan ke LLM:** Menempelkan seluruh output CMD (`git init` → `git add` → `git commit` → `git push`) ke LLM.
- **Analisis & Solusi Kelompok:** LLM mendeteksi semua file berada di subfolder `midi-generator/midi-generator/` karena zip diekstrak ke dalam folder bernama sama. Akibatnya README tidak tampil di halaman depan GitHub. Solusi: memindahkan file ke root dengan `git mv midi-generator/<file> .` lalu commit baru. Pelajaran: selalu periksa path di output `git commit` dan cek isi folder dengan `dir` sebelum push.

**Eror 2.2 — Autentikasi browser tidak terbuka saat push**
- **Pesan Eror dari Konsol:**
  ```
  C:\...\midi-generator>git push -u origin main
  info: please complete authentication in your browser...
  ```
  (proses menggantung, browser tidak terbuka)
- **Prompt Perbaikan ke LLM:** "info: please complete authentication in your browser... ini ga kebukaa" lalu "cara batalinnya bisa?"
- **Analisis & Solusi Kelompok:** Git Credential Manager di Windows gagal membuka browser otomatis. Proses dibatalkan dengan `Ctrl + C`, lalu mode autentikasi diganti ke kode perangkat dengan `git config --global credential.gitHubAuthModes device`. Setelah itu push berhasil:
  ```
   * [new branch]      main -> main
  branch 'main' set up to track 'origin/main'.
  ```
  Pelajaran: membatalkan push tidak menghapus commit lokal; status push bisa dicek dengan `git status` (harus ada "up to date with 'origin/main'").

**Kendala 2.3 — Repo tidak ditemukan (404) di browser**
- **Pesan:** "Page not found" saat membuka `github.com/AinunWidyah/midi-generator`.
- **Analisis & Solusi Kelompok:** Repo bersifat Private sehingga GitHub menampilkan 404 untuk akun yang tidak punya akses. Penyebabnya, link dibuka di browser yang tidak login dengan akun pemilik repo. Solusi: membuka di browser yang sudah login dengan akun yang benar.

**Eror 2.4 — Push ditolak (fetch first)**
- **Pesan Eror dari Konsol:**
  ```
  C:\...\midi-generator>git push
  To https://github.com/AinunWidyah/midi-generator.git
   ! [rejected]        main -> main (fetch first)
  error: failed to push some refs to 'https://github.com/AinunWidyah/midi-generator.git'
  hint: Updates were rejected because the remote contains work that you do not
  hint: have locally.
  ```
- **Prompt Perbaikan ke LLM:** Menempelkan output lengkap `git status` → `git add docs/` → `git commit` → `git push` yang gagal.
- **Analisis & Solusi Kelompok:** GitHub sudah memiliki commit yang belum ada di laptop (README diedit lewat website untuk mengisi nama anggota). Git menolak push agar perubahan itu tidak tertimpa. Solusi: `git pull --no-rebase --no-edit` (menggabungkan perubahan tanpa membuka editor teks), lalu `git push` ulang:
  ```
  Merge made by the 'ort' strategy.
   README.md | 12 ++++++------
  ...
     a7dd038..1218d13  main -> main
  ```
  Pelajaran: biasakan `git pull` sebelum mulai bekerja dan sebelum push, terutama karena empat anggota bekerja di repo yang sama.
