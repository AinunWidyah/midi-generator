# LOG INTERAKSI LLM - KELOMPOK [Nama Kelompok]

- **Tanggal Sesi:** 21/10/2026 & 02/11/2026
- **Anggota / Peran:** Atha Imtinan U.L — Data Pipeline & Preprocessing Engineer (Anggota 2)
- **Target Modul:** MIDI Tokenizer (src/tokenizer.py), audit dataset MIDI, dan augmentasi transposisi data train (src/dataset.py)
- **Tautan Sesi Chat:** (https://chatgpt.com/share/6ae8518a-a194-83ec-82f2-e37b72c526fb & https://claude.ai/share/9bedf079-867f-433b-bb43-95f8e678be4b)

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)

- **Prompt Kelompok:**

"Berdasarkan kontrak modul yang telah disepakati kelompok, bantu buat implementasi src/tokenizer.py untuk proyek MIDI String Generator. Implementasi harus menggunakan pretty_midi untuk membaca dan menulis file MIDI, mengikuti format token yang telah ditentukan, serta menggunakan build_vocab() dari src/config.py agar ID token konsisten. Buat fungsi encode(), decode(), tokens_to_ids(), ids_to_tokens(), dan transpose(). Pastikan implementasi dapat menangani token tidak valid tanpa menyebabkan program langsung berhenti."

- **Alasan Khusus:**

Prompt ini digunakan agar implementasi tokenizer mengikuti kontrak modul kelompok dan tidak membuat format token atau pemetaan ID secara terpisah. Fungsi tokenizer juga perlu disiapkan agar dapat digunakan oleh modul dataset dan proses pelatihan model. Penggunaan pretty_midi dipilih karena merupakan library yang telah disepakati untuk proses input dan output MIDI.

- **Output Model:**

LLM menghasilkan rancangan kode tokenizer.py dengan fungsi-fungsi berikut:

encode(midi_path): Membaca file MIDI, memilih track instrumen non-drum dengan melodi monofonik, kemudian mengubah notasi menjadi urutan token.
decode(tokens, out_path, tempo=120.0): Mengubah urutan token menjadi file MIDI.
tokens_to_ids(tokens): Mengubah token berbentuk string menjadi ID berdasarkan vocabulary dari build_vocab().
ids_to_tokens(ids): Mengubah ID kembali menjadi token string.
transpose(tokens, semitones): Melakukan transposisi nada dengan menggeser pitch sesuai jumlah semitone.

Format token mengikuti kontrak kelompok, yaitu <BOS>, token NOTE_{pitch} atau REST, token DUR_{1..16}, dan diakhiri <EOS>. Kode juga dilengkapi pemeriksaan mandiri sederhana serta pengecekan token yang tidak valid pada proses decoding.

- **Prompt Audit Dataset:**

"Bantu analisis hasil audit dataset MIDI. Periksa apakah seluruh file dapat dibaca, apakah terdapat file kosong, dan apakah setiap file memiliki track melodi monofonik. Identifikasi file yang tidak memenuhi asumsi kontrak tokenizer agar dapat dipisahkan dari dataset utama tanpa menghapus file sumber."

- **Alasan Khusus:**

Audit dilakukan untuk memastikan dataset yang digunakan sesuai dengan asumsi tokenizer, khususnya format melodi monofonik. File yang tidak memenuhi asumsi tersebut perlu dipisahkan agar tidak menimbulkan masalah pada proses tokenisasi dan pelatihan model.

- **Analisis & Solusi Kelompok:**

LLM membantu menyusun pendekatan audit dataset untuk memeriksa keterbacaan file MIDI, keberadaan not, serta karakteristik track instrumen.

Hasil audit menunjukkan:

Total file MIDI: 1.034
File terbaca dan memiliki not: 1.034
File kosong: 0
File gagal dibaca: 0
File yang tidak memiliki track melodi monofonik: 22
File dengan kandidat track yang menghasilkan lebih dari 256 token: 775

Berdasarkan hasil tersebut, kelompok memutuskan untuk memisahkan 22 file yang tidak memenuhi asumsi monofonik dari dataset aktif. File tidak dihapus, melainkan dipindahkan ke folder _excluded_non_monophonic agar tetap dapat digunakan jika diperlukan pada pengembangan berikutnya.

Setelah proses pemisahan, dilakukan verifikasi ulang dan diperoleh:

Dataset utama: 1.012 file MIDI
Folder pengecualian: 22 file MIDI

Pelajaran yang diperoleh adalah validasi dataset perlu dilakukan sebelum proses tokenisasi dan pelatihan model agar format data yang masuk sesuai dengan asumsi desain tokenizer.

### 2. TAHAP OPTIMASI TOKENIZER (21/10/2026)

- **Prompt:**

"Periksa kembali rancangan tokenizer agar sesuai dengan kontrak modul. Pastikan vocabulary diambil dari build_vocab(), durasi not dikuantisasi ke satuan enam belas not, jeda antar-not direpresentasikan menggunakan REST dan DUR, serta fungsi decoding dapat melewati token yang tidak valid tanpa menghentikan proses."

- **Alasan Khusus:**

Tahap ini dilakukan untuk memastikan rancangan tokenizer konsisten dengan kontrak modul dan dapat digunakan sebagai penghubung antara data MIDI, dataset, dan model.

- **Hasil / Perubahan:**

Optimasi berfokus pada kesesuaian implementasi terhadap kontrak, bukan pada peningkatan metrik model.

Perubahan dan ketentuan yang diperhatikan:

Vocabulary diambil dari build_vocab() agar pemetaan token dan ID konsisten dengan konfigurasi proyek.
Durasi not dikonversi ke satuan enam belas not (sixteenth-note units) dan dibatasi pada rentang 1–16.
Jeda positif antar-not direpresentasikan menggunakan token REST yang diikuti token durasi.
Urutan token diawali <BOS> dan diakhiri <EOS>.
Proses decoding dirancang untuk melewati token yang tidak valid agar tidak langsung menyebabkan program berhenti.
Fungsi transposisi membatasi pitch pada rentang MIDI 0–127.

Kode awal tokenizer.py berhasil dibuat dan lolos pemeriksaan sintaks Python. Namun, kompatibilitas dengan implementasi aktual src/config.py serta pengujian round-trip MIDI (decode(encode(file))) masih perlu diverifikasi langsung di lingkungan proyek.

### 3. TAHAP INISIASI AUGMENTASI TRANSPOSISI (02/11/2026)

Target: Implementasi augmentasi transposisi pada src/dataset.py berdasarkan temuan K1 dalam docs/REVIEW_INTEGRASI_01.md.

- **Prompt Kelompok:**

"Tolong tambahkan augmentasi transposisi (K1 di docs/REVIEW_INTEGRASI_01.md) di dataset.py: geser nada acak −5..+5 semiton, khusus data train. Data kita kecil jadi rawan overfitting. Lewat branch + PR ya."

Disusul pertanyaan klarifikasi:

"Ini maksudnya kaya gimana?"
"Ini diganti di dataset.py atau gimana?"

Setelah itu, isi dataset.py dikirimkan kepada LLM untuk dianalisis dan disesuaikan dengan implementasi yang sudah ada.

- **Alasan Khusus:**

Jumlah data train yang relatif kecil berpotensi menyebabkan overfitting. Augmentasi transposisi digunakan agar model dapat melihat variasi tinggi nada dari lagu yang sama tanpa mengubah pola melodinya.

Augmentasi hanya diterapkan pada data train, sedangkan validation tetap menggunakan data asli. Hal ini dilakukan agar evaluasi model tetap merepresentasikan data tanpa augmentasi sehingga metrik validasi dapat dibandingkan secara adil.

- **Output Model:**

LLM menyarankan tiga perubahan utama pada MIDIStringDataset dan create_dataloaders:

Menambahkan parameter baru pada __init__:
augment: bool = False
max_shift: int = 5
Menambahkan proses augmentasi pada __getitem__, sebelum chunk dipecah menjadi input_ids dan labels:
python
if self.augment:
    shift = random.randint(-self.max_shift, self.max_shift)
    if shift != 0:
        tokens = tokenizer.ids_to_tokens(chunk)
        tokens = tokenizer.transpose(tokens, shift)
        chunk = tokenizer.tokens_to_ids(tokens)

Mengaktifkan augmentasi hanya pada dataset train:
train_dataset menggunakan augment=True.
val_dataset tetap menggunakan nilai default sehingga tidak mengalami augmentasi.

Dengan pendekatan tersebut, setiap pengambilan sampel data train dapat memperoleh variasi transposisi secara acak dalam rentang −5 hingga +5 semitone.

### 4. TAHAP PENANGANAN ERROR / DEBUGGING (02/11/2026)

- **Pesan Error dari Notebook / Review Kode:**

Saat melakukan review terhadap branch feat-transpose-augmentation pada commit 3c7fc5a, ditemukan error berikut:

python
File "dataset.py", line 176
    chunk = tokenizer.tokens_to_ids(tokens)chunk = self.examples[index]
                                           ^^^^^
SyntaxError: invalid syntax


Error tersebut ditemukan sebelum proses merge. Karena train.py mengimpor dataset.py, kesalahan sintaks ini dapat merusak seluruh alur training jika kode di-merge tanpa perbaikan.

- **Prompt Perbaikan ke LLM:**

Catatan hasil review yang berisi pesan SyntaxError dan saran penggunaan TrainConfig.transpose_range ditempelkan ke percakapan. Selanjutnya, isi dataset.py setelah diperbaiki dikirimkan kembali untuk diperiksa.

- **Analisis & Solusi Kelompok:**

Hasil analisis menunjukkan bahwa akar masalah berasal dari kesalahan salin-tempel kode. Blok pengganti dari LLM diawali dengan baris chunk = self.examples[index], tetapi baris tersebut tertempel di ujung baris lain sehingga dua baris menyatu.

- **Perbaikan yang dilakukan:**

Menghapus sisa chunk = self.examples[index] yang menempel pada baris 176 sehingga baris tersebut hanya berisi:
python
chunk = tokenizer.tokens_to_ids(tokens)

Mengganti nilai manual max_shift=5 menjadi:
python
max_shift=config.transpose_range


Tujuannya agar rentang transposisi diatur melalui konfigurasi proyek di config.py, bukan ditulis secara manual di dataset.py.

Melakukan pemeriksaan kode sebelum push menggunakan:
bash
python -m py_compile src/dataset.py
python src/dataset.py


Perintah py_compile berhasil dijalankan tanpa menghasilkan output error.

- **Kendala Git dan Pull Request:**

Selain kesalahan sintaks, ditemukan kendala pada proses Git. Setelah langkah penghubung repository dijalankan ulang menggunakan git branch -M main dan git reset origin/main, branch lokal berganti nama menjadi main. Akibatnya, proses git push ditolak karena upstream branch tidak sesuai.

Solusi yang dilakukan adalah tidak memaksakan push, tetapi membuat branch baru bernama fix-dataset-transpose-range, kemudian melakukan push melalui branch tersebut dan membuka Pull Request.

- **Hasil pemeriksaan menggunakan:**

bash
git diff origin/main HEAD


menunjukkan bahwa baris yang rusak ternyata sudah terdapat pada branch main. Oleh karena itu, perbaikan PR perlu segera di-merge dan anggota kelompok lain diberi tahu agar tidak menjalankan train.py dari main sebelum perbaikan diterapkan.

- **Pelajaran yang Diperoleh:**

Saat menerapkan perubahan kode, pastikan baris lama yang diganti dihapus terlebih dahulu agar tidak terjadi penggabungan dua baris secara tidak sengaja.
Selalu jalankan py_compile sebelum melakukan commit untuk mendeteksi kesalahan sintaks.
Proses code review sebelum merge sangat penting karena dapat menemukan error yang menyebabkan branch utama tidak dapat dijalankan.
Langkah penghubung repository cukup dilakukan satu kali dan tidak perlu diulang tanpa alasan.
Selalu periksa hasil git diff sebelum melakukan push untuk memastikan perubahan yang dikirimkan sesuai dengan tujuan.
Gunakan konfigurasi terpusat seperti config.transpose_range agar parameter augmentasi lebih mudah diubah dan konsisten dengan struktur proyek.

### 5. TAHAP OPTIMASI AUGMENTASI (02/11/2026)

- **Prompt:**

Tidak ada prompt optimasi khusus pada tahap ini. Implementasi augmentasi transposisi itu sendiri merupakan upaya untuk mengurangi risiko overfitting pada dataset train yang relatif kecil.

- **Hasil / Perubahan:**

Perubahan kode difokuskan pada implementasi augmentasi transposisi di src/dataset.py.

- **Rangkuman perubahan:**

Augmentasi dilakukan secara acak dengan rentang −5 hingga +5 semitone.
Augmentasi hanya diterapkan pada dataset train.
Dataset validation tetap menggunakan data asli tanpa augmentasi.
Parameter rentang transposisi diambil dari config.transpose_range.
Perbaikan sintaks dilakukan agar file dataset.py dapat diimpor dan dijalankan.
Perubahan tercatat pada commit perbaikan 37d177a dengan ringkasan perubahan src/dataset.py (+2/−1 baris).

- **Perubahan Metrik:**

Belum diukur. Perbandingan validation loss sebelum dan sesudah penerapan augmentasi masih perlu dilakukan setelah proses training ulang. Dengan demikian, efektivitas augmentasi terhadap pengurangan overfitting belum dapat disimpulkan secara kuantitatif.

### KESIMPULAN

LLM digunakan sebagai alat bantu dalam pengembangan data pipeline dan preprocessing proyek MIDI String Generator, mulai dari perancangan tokenizer, audit dataset MIDI, hingga implementasi augmentasi transposisi pada dataset train.

Pada sesi 21/10/2026, LLM membantu merancang implementasi src/tokenizer.py yang mencakup fungsi encoding, decoding, konversi token ke ID, konversi ID ke token, dan transposisi nada. Selain itu, proses audit dataset berhasil mengidentifikasi 22 file MIDI yang tidak memenuhi asumsi melodi monofonik. File tersebut dipisahkan ke folder _excluded_non_monophonic tanpa dihapus sehingga dataset utama berisi 1.012 file MIDI yang lebih sesuai dengan asumsi tokenizer.

Pada sesi 02/11/2026, LLM membantu mengimplementasikan augmentasi transposisi pada src/dataset.py berdasarkan temuan K1 dalam dokumen review integrasi. Augmentasi dilakukan dengan menggeser pitch secara acak dalam rentang −5 hingga +5 semitone dan hanya diterapkan pada data train untuk menambah variasi data serta mengurangi risiko overfitting. Data validation tetap dipertahankan tanpa augmentasi agar evaluasi model tetap objektif.

Dalam proses implementasi, ditemukan kesalahan sintaks akibat kesalahan salin-tempel kode yang kemudian diperbaiki melalui proses debugging dan code review. Selain itu, kendala Git juga diselesaikan dengan membuat branch perbaikan baru dan menggunakan Pull Request agar perubahan dapat ditinjau sebelum diterapkan ke branch utama.

Secara keseluruhan, kedua sesi menunjukkan bahwa LLM membantu mempercepat proses pengembangan, pemeriksaan, dan perbaikan kode. Namun, hasil implementasi tetap memerlukan verifikasi langsung di lingkungan proyek. Beberapa pengujian lanjutan yang masih perlu dilakukan meliputi:

Verifikasi kompatibilitas tokenizer dengan build_vocab() pada src/config.py.
Pengujian round-trip MIDI menggunakan decode(encode(file)).
Pengujian integrasi tokenizer dengan modul dataset dan model.
Verifikasi bahwa augmentasi hanya aktif pada data train.
Perbandingan validation loss sebelum dan sesudah penerapan augmentasi untuk mengetahui dampaknya terhadap overfitting.

Dengan demikian, pengembangan tokenizer dan augmentasi transposisi telah memiliki implementasi awal serta proses debugging yang terdokumentasi, sementara pengujian integrasi dan evaluasi performa akhir masih menjadi tahap lanjutan sebelum modul dinyatakan selesai.