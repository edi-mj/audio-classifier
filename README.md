# Audio Classifier - Pengenalan Perintah Suara

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.51.0-FF4B4B.svg)
![Librosa](https://img.shields.io/badge/librosa-0.11.0-orange.svg)
![NumPy](https://img.shields.io/badge/numpy-2.3.4-013243.svg)

Aplikasi web interaktif untuk mengenali perintah suara "Buka" dan "Tutup" menggunakan algoritma Random Forest. Dibangun dengan Streamlit dan Librosa untuk memberikan pengalaman pengguna yang sederhana dan responsif.

## Demo Aplikasi

Coba aplikasi secara langsung di: [audio-classifier.streamlit.app](https://audio-classifier.streamlit.app/)

## Deskripsi Proyek

Audio Classifier adalah sistem klasifikasi audio berbasis machine learning yang mampu membedakan perintah suara "buka" dan "tutup". Aplikasi ini menggunakan teknik ekstraksi fitur audio yang canggih dan algoritma Random Forest Classifier untuk memberikan prediksi real-time dengan tingkat kepercayaan yang tinggi.

### Fitur Utama

- **Perekaman Audio Real-time** - Rekam suara langsung dari browser tanpa perlu alat tambahan
- **Preprocessing Otomatis** - Trimming keheningan dan normalisasi audio secara otomatis
- **Ekstraksi Fitur Audio** - Menggunakan 4 fitur statistik:
  - Zero-Crossing Rate (ZCR)
  - Root Mean Square Energy (RMS)
  - Spectral Centroid
  - Spectral Bandwidth
- **Prediksi dengan Confidence Score** - Menampilkan tingkat kepercayaan model terhadap prediksi
- **Interface User-Friendly** - Antarmuka web yang bersih dan mudah digunakan

## Tech Stack

**Frontend & Framework:**

- [Streamlit](https://streamlit.io/) - Framework aplikasi web untuk machine learning
- [streamlit-audiorec](https://github.com/stefanrmmr/streamlit-audio-recorder) - Komponen perekaman audio

**Audio Processing & ML:**

- [Librosa](https://librosa.org/) - Library analisis dan pemrosesan audio
- [NumPy](https://numpy.org/) - Komputasi numerik dan array processing
- [SoundFile](https://pysoundfile.readthedocs.io/) - Reading dan writing file audio
- [scikit-learn](https://scikit-learn.org/) - Random Forest Classifier dan preprocessing
- [Joblib](https://joblib.readthedocs.io/) - Model serialization

## Instalasi

### Prasyarat

- Python 3.8 atau lebih baru
- pip (Python package manager)

### Langkah Instalasi

1. Clone repository ini

```bash
git clone https://github.com/edi-mj/audio-classifier.git
cd audio-classifier
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Pastikan model sudah tersedia
   - File model `audio_classifier.pkl` harus ada di direktori root proyek
   - Jika belum ada, jalankan script training terlebih dahulu

## Cara Penggunaan

1. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

2. Aplikasi akan terbuka di browser (biasanya di `http://localhost:8501`)

3. Klik tombol rekam untuk memulai perekaman

4. Ucapkan perintah "buka" atau "tutup" dengan jelas

5. Klik tombol stop untuk menghentikan perekaman

6. Aplikasi akan menampilkan hasil prediksi beserta confidence score

## Cara Kerja

### 1. Preprocessing Audio

Audio yang direkam akan melalui tahap preprocessing:

- **Trimming** - Menghilangkan bagian keheningan di awal dan akhir
- **Normalisasi** - Menyeimbangkan amplitudo audio

### 2. Ekstraksi Fitur

Dari audio yang sudah diproses, sistem mengekstrak 8 fitur statistik (mean dan std dari 4 metrik):

- Zero-Crossing Rate untuk mendeteksi frekuensi perubahan sinyal
- RMS Energy untuk mengukur energi audio
- Spectral Centroid untuk menangkap "center of mass" dari spektrum
- Spectral Bandwidth untuk mengukur lebar spektrum frekuensi

### 3. Prediksi

Fitur yang diekstrak akan dinormalisasi menggunakan scaler, kemudian dikirim ke model Random Forest Classifier untuk mendapatkan prediksi dan confidence score. Random Forest dipilih karena kemampuannya menangani fitur audio dengan baik dan memberikan probabilitas prediksi yang reliable.

## Struktur File

```
audio-classifier/
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Dependencies Python
├── audio_classifier.pkl    # Model terlatih (tidak termasuk di repo)
└── README.md              # Dokumentasi proyek
```

## Konfigurasi

Parameter yang dapat disesuaikan di `app.py`:

- `TARGET_SR = 22050` - Sample rate untuk audio processing
- `SILENCE_THRESHOLD = 30` - Threshold (dalam dB) untuk trimming keheningan

## Kontribusi

Kontribusi selalu diterima dengan tangan terbuka. Untuk berkontribusi:

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## Troubleshooting

**Model tidak ditemukan**

- Pastikan file `audio_classifier.pkl` ada di direktori yang sama dengan `app.py`
- Jalankan script training model terlebih dahulu

**Audio tidak terdeteksi**

- Coba bicara lebih keras atau lebih dekat dengan mikrofon
- Periksa permission mikrofon di browser
- Kurangi nilai `SILENCE_THRESHOLD` jika audio terus ter-trim

**Error saat instalasi librosa**

- Librosa memerlukan FFmpeg atau libsndfile. Install sesuai OS Anda
- Windows: `pip install librosa[sound]`
- Linux: `sudo apt-get install libsndfile1`
- macOS: `brew install libsndfile`

---

Dibuat dengan Python dan Streamlit
