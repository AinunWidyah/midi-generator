# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 21/10/2026
- **Anggota / Peran:** Ainun Widyah Astanti — Prompt Engineer & LLM Navigator (Ketua)
- **Target Modul:** Keputusan teknis Ketua — aturan lokasi data (Kontrak v1.1), modul `preprocessing.py` (Kontrak v1.2), dan verifikasi Asumsi 1
- **Tautan Sesi Chat:** [tempel share link chat ini]

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI

**Keputusan 1 — Lokasi data (Kontrak v1.1)**
- **Prompt Kelompok:** "untuk anggota 2 telah mendapatkan dataset .mid, dimanakah dataset tersebut di letakkan kemudian dimana diletakkan hasil data pre processing"
- **Alasan Khusus:** Anggota 2 sudah memegang dataset, tetapi kontrak v1.0 belum mengatur lokasi file. Tanpa aturan, setiap anggota bisa memakai path berbeda dan kode tidak jalan saat digabung (terutama antara Windows dan Colab).
- **Output Model & Keputusan:**
  - Data mentah di `data/raw/<dataset_name>/` (tidak boleh diubah), hasil preprocessing di `data/processed/<dataset_name>/`.
  - Path diambil dari properti `TrainConfig` (`raw_dir`, `processed_dir`, `tokens_path`, dll.), tidak ditulis manual.
  - Isi `data/` tidak di-commit; hanya `data/README.md` (sumber dataset) yang masuk repo, dengan `.gitignore` diberi pengecualian `!data/README.md`.

**Keputusan 2 — Folder `melody` langsung di `data/`?**
- **Prompt Kelompok:** "untuk di folder data langsung folder melody yang berisi .mid apakah bisa?"
- **Analisis & Keputusan:** Ditolak, karena `config.py` mencari `data/raw/nottingham/`. Solusi tanpa mengubah kontrak: isi folder `MIDI/melody` dari dataset Nottingham disalin ke `data/raw/nottingham/` (hanya `*.mid`), dan 22 file di folder karantina dikembalikan agar data mentah utuh 1.034 file.

**Keputusan 3 — Modul `src/preprocessing.py` (Kontrak v1.2)**
- **Prompt Kelompok:** "apakah tambah src/preprocessing.py, sama catat keputusan asumsi 1 di log di lakukan sekaligus anggota 2" lalu "bolehh"
- **Alasan Khusus:** Anggota 2 sudah membuat `filter_midi_monophonic.py` di luar repo, dan skrip itu **memindahkan** file dari data mentah. Perlu aturan resmi tentang tempat dan perilaku skrip preprocessing sebelum Anggota 2 melanjutkan.
- **Pembagian tugas yang disepakati:** Ketua menulis **aturan** (kontrak), Anggota 2 menulis **kode** lewat sesi vibe coding sendiri. Keputusan Ketua dicatat di log ini; proses teknis dicatat di log Anggota 2.
- **Output Model & Keputusan:**
  - File baru `src/preprocessing.py` (milik Anggota 2), dijalankan dengan `python src/preprocessing.py`.
  - Lima langkah wajib: filter monofonik → tokenisasi → split 90/10 per lagu (seed 42) → statistik & cek asumsi 2–3 → ringkasan.
  - **Tidak boleh** memindahkan/menghapus file di `raw/`; file yang dikecualikan cukup dicatat di `excluded_midi_report.csv` (kolom `file,alasan,detail`).
  - Format `stats.json` dibakukan agar Ketua bisa membaca hasil cek asumsi dengan cara yang sama setiap kali.
  - `config.py` ditambah properti `excluded_path`.

### 2. VERIFIKASI ASUMSI 1 (berdasarkan laporan Anggota 2)

- **Data dari Anggota 2:** output skrip `filter_midi_monophonic.py` pada folder `nottingham-dataset-master/MIDI/melody`:
  ```
  Total file diperiksa              : 1034
  File yang akan dikecualikan       : 22
  File yang dipertahankan           : 1012
  File gagal dibaca (perlu ditinjau): 0
  File dipindahkan ke karantina: 22
  ```
- **Batas aman di kontrak:** file non-monofonik < 5%.
- **Hasil:** 22 / 1.034 = **2,1%** → di bawah batas.
- **Keputusan Ketua:**
  1. Asumsi 1 dinyatakan **terverifikasi**. Format token v1 (melodi monofonik) **dipertahankan**, `config.py` dan `model.py` **tidak diubah**.
  2. 22 file tersebut **dikecualikan** dari training, dicatat di `excluded_midi_report.csv`, tetapi file aslinya tetap disimpan di `raw/`.
  3. Asumsi 2 (durasi 1–16) dan 3 (panjang lagu) ditunda sampai `tokenizer.encode` selesai, karena butuh hasil tokenisasi.

### 3. TAHAP PENANGANAN EROR / CATATAN PERBAIKAN

**Catatan 3.1 — Skrip filter mengubah data mentah**
- **Temuan:** `filter_midi_monophonic.py` memindahkan 22 file ke folder karantina, sehingga data mentah tidak lagi utuh dan anggota lain yang mengunduh dataset akan mendapat isi berbeda.
- **Perbaikan:** dibakukan di Kontrak v1.2 bahwa `raw/` tidak boleh diubah; pengecualian hanya dicatat di CSV dan dilewati oleh `preprocessing.py`.

**Catatan 3.2 — Skrip berada di luar repository**
- **Temuan:** skrip dan dataset berada di `OneDrive\Documents\KULIAH\...`, di luar folder proyek, sehingga tidak tercatat sebagai kontribusi di Git dan `cfg.raw_dir` tidak menemukan data.
- **Perbaikan:** dataset disalin ke `data/raw/nottingham/`; logika skrip dipindahkan ke `src/preprocessing.py` dan di-commit oleh Anggota 2.

**Catatan 3.3 — Nama file berbeda dengan kontrak**
- **Temuan:** setelah Kontrak v1.2 ditulis dengan nama `preprocess.py`, ternyata Anggota 2 sudah lebih dulu membuat file bernama `preprocessing.py`.
- **Prompt ke LLM:** "apakah perlu di tambahkan, dan juga preprocessing.py sudah terlanjur dibuat anggota 2"
- **Perbaikan:** kontrak **menyesuaikan nama file Anggota 2** (`src/preprocessing.py`) agar tidak ada dua file dengan fungsi sama; file kosong `preprocess.py` dari Ketua tidak jadi ditambahkan. Isi file Anggota 2 tetap akan di-review terhadap aturan kontrak (terutama: tidak boleh memindahkan file di `raw/`).

**Pelajaran:** kontrak perlu diperbarui secara bertahap mengikuti temuan di lapangan (v1.0 → v1.1 → v1.2), dan setiap perubahan dicatat beserta alasannya.
