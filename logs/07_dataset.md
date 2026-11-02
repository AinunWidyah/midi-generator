# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 21/10/2026
- **Anggota / Peran:** Atha Imtinan U.L — Data Pipeline & Preprocessing Engineer (Anggota 2)
- **Target Modul:** Persiapan Dataset (`src/dataset.py`, Audit, Validasi, dan Preprocessing Dataset MIDI (Nottingham MIDI Dataset)
- **Tautan Sesi Chat:** (https://chatgpt.com/share/6ae8518a-a194-83ec-82f2-e37b72c526fb)

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)

- **Prompt Kelompok:**

  "Saya bertanggung jawab sebagai Member 2 — Data Pipeline & Preprocessing Engineer pada proyek MIDI String Generator. Saya ingin melakukan audit terhadap dataset Nottingham MIDI untuk mengetahui jumlah file MIDI, file yang berhasil dibaca, file kosong, file yang mengalami error, serta file yang memiliki track monofonik dan polifonik. Tolong bantu buatkan script Python menggunakan library `pretty_midi` untuk melakukan pemeriksaan tersebut."

- **Alasan Khusus:**

  Audit dataset dilakukan sebelum proses preprocessing dan pembuatan tokenizer agar kualitas data dapat diketahui terlebih dahulu. Pemeriksaan monofonik diperlukan untuk mengidentifikasi file yang sesuai dengan pendekatan pemodelan melodi tunggal. Selain itu, pemeriksaan dilakukan untuk memastikan file MIDI dapat dibaca dan memiliki informasi nada yang dapat diproses lebih lanjut.

- **Output Model:**

  Model memberikan script Python `audit_midi.py` menggunakan library `pretty_midi`.

  Script tersebut dikembangkan untuk:
  - Membaca seluruh file berekstensi `.mid` dan `.midi`.
  - Menghitung jumlah file MIDI yang ditemukan.
  - Memeriksa file yang berhasil dibaca dan memiliki data nada.
  - Mengidentifikasi file kosong dan file yang gagal dibaca.
  - Memeriksa karakteristik monofonik dan polifonik pada track MIDI.
  - Menghasilkan laporan audit dalam format CSV.

  Pada pengembangan berikutnya, script diperluas menjadi dua laporan:
  - `audit_midi_summary.csv`: ringkasan hasil audit setiap file MIDI.
  - `audit_midi_tracks.csv`: informasi hasil pemeriksaan pada masing-masing track.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

- **Pesan Eror dari Notebook:**

  Tidak terdapat pesan error eksekusi yang dilaporkan selama proses audit. Script berhasil dijalankan melalui PowerShell pada direktori dataset MIDI.

- **Prompt Perbaikan ke LLM:**

  "Saya sudah menjalankan audit dataset MIDI. Tolong bantu menganalisis hasil CSV untuk mengetahui berapa banyak file yang memiliki track monofonik, berapa banyak yang tidak memiliki track monofonik, serta apakah durasi nada dan estimasi panjang token sesuai dengan kebutuhan model."

- **Analisis & Solusi Kelompok:**

  Model membantu menganalisis hasil audit dan menemukan beberapa informasi penting:

  1. Terdapat **1.034 file MIDI** pada dataset awal.
  2. Seluruh 1.034 file berhasil dibaca dan memiliki data nada.
  3. Tidak ditemukan file kosong maupun file yang gagal dibaca.
  4. Sebanyak **1.012 file** memiliki setidaknya satu track monofonik yang dapat dipertimbangkan untuk preprocessing.
  5. Sebanyak **22 file** tidak memiliki track monofonik yang memenuhi kriteria pemeriksaan.
  6. Sebagian besar kandidat track memiliki durasi nada dalam rentang 1–16 satuan not seperenambelas.
  7. Sebanyak **775 kandidat track** memiliki estimasi panjang token lebih dari `block_size=256`.

  Berdasarkan hasil tersebut, diputuskan untuk memisahkan 22 file yang tidak memiliki track monofonik dari dataset utama.

  Model kemudian membantu membuat script `filter_midi_monophonic.py` untuk melakukan proses pemisahan secara otomatis. File yang tidak memenuhi kriteria dipindahkan ke folder karantina `_excluded_non_monophonic/`, bukan dihapus permanen. Script juga menghasilkan laporan `excluded_midi_report.csv`.

  Setelah proses filtering selesai, jumlah dataset utama berhasil diverifikasi menjadi **1.012 file MIDI**, sedangkan 22 file lainnya berada di folder karantina.

  **Pelajaran yang diperoleh:** proses audit perlu dilakukan sebelum preprocessing agar data yang digunakan sesuai dengan kebutuhan model. File yang tidak memenuhi kriteria sebaiknya dipisahkan terlebih dahulu, bukan langsung dihapus, sehingga masih dapat diperiksa kembali apabila diperlukan.

### 3. TAHAP OPTIMASI

- **Prompt:**

  "Dari hasil audit, terdapat 775 kandidat track dengan estimasi jumlah token lebih dari 256. Apakah file tersebut harus dikeluarkan dari dataset, atau ada pendekatan lain agar tetap dapat digunakan oleh model?"

- **Hasil / Perubahan Metrik:**

  Model menjelaskan bahwa `block_size=256` merupakan batas panjang konteks yang digunakan model, sehingga file dengan estimasi token lebih panjang tidak harus langsung dikeluarkan dari dataset.

  Pendekatan yang disarankan adalah menggunakan pemotongan sequence atau windowing agar bagian-bagian dari lagu yang panjang tetap dapat dimanfaatkan dalam proses training.

  Ringkasan perubahan dataset:

  | Keterangan | Jumlah |
  |---|---:|
  | Dataset awal | 1.034 file |
  | File berhasil dibaca dan memiliki nada | 1.034 file |
  | File tanpa track monofonik | 22 file |
  | File yang dipisahkan ke folder karantina | 22 file |
  | Dataset utama setelah filtering | **1.012 file** |
  | Kandidat track dengan estimasi token > 256 | 775 |

  **Catatan:** jumlah estimasi token belum merupakan hasil tokenisasi final. Validasi panjang sequence perlu dilakukan kembali setelah tokenizer diimplementasikan.

### 4. TAHAP LANJUTAN (Persiapan Tokenizer dan Dataset Loader)

- **Target Berikutnya:**

  Melanjutkan implementasi modul preprocessing dengan membuat `src/tokenizer.py` dan `src/dataset.py`.

- **Rencana Pengerjaan:**

  1. Membuat tokenizer untuk mengubah informasi nada dan durasi dari MIDI menjadi representasi token.
  2. Menyiapkan fungsi encoding dan decoding.
  3. Melakukan pengujian round-trip untuk memastikan hasil decoding dapat merepresentasikan kembali data MIDI yang telah diencode.
  4. Membuat dataset loader untuk membaca data hasil preprocessing.
  5. Menyiapkan pasangan input dan target `(x, y)` untuk training.
  6. Menerapkan padding atau windowing sesuai kebutuhan `block_size=256`.
  7. Membagi dataset menjadi data training dan validation dengan rasio 90:10 berdasarkan lagu.

- **Status:** Audit dan filtering dataset selesai. Implementasi tokenizer dan dataset loader menjadi tahap pengerjaan berikutnya.