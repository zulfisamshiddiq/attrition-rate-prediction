# Proyek Akhir: Menyelesaikan Permasalahan HR di Jaya Jaya Maju

## 1. Business Understanding

Jaya Jaya Maju adalah perusahaan multinasional dengan lebih dari 1000 karyawan. Saat ini, perusahaan menghadapi masalah tingginya _attrition rate_ (rasio karyawan yang keluar) hingga melebihi 10%. Manajer HR ingin mengidentifikasi faktor-faktor utama penyebab tingginya tingkat _attrition_ ini untuk mencegah masalah yang lebih parah dan menjaga stabilitas operasional perusahaan.

## 2. Permasalahan Bisnis

Berdasarkan kondisi di atas, pertanyaan bisnis yang ingin dijawab melalui proyek ini adalah:

- Apa faktor utama yang mendorong karyawan untuk meninggalkan perusahaan (_resign_)?
- Bagaimana cara memantau faktor-faktor tersebut secara efektif oleh tim HR?
- Langkah preventif apa yang bisa diambil untuk menekan _attrition rate_ di bawah 10%?

## 3. Cakupan Proyek

Proyek ini bertujuan untuk membantu perusahaan Jaya Jaya Maju dalam mengatasi tingginya tingkat keluar-masuk karyawan (attrition). Melalui pendekatan Data Science, proyek ini merangkum proses Exploratory Data Analysis (EDA) untuk menemukan akar masalah, pembuatan Business Dashboard interaktif untuk pemantauan HR, serta implementasi model Machine Learning untuk memprediksi probabilitas resign karyawan di masa depan.

## 4. Sumber Data

Dataset yang digunakan dalam proyek ini berisi metrik demografi dan performa karyawan yang dapat diakses melalui tautan berikut:

**Link dataset:** [Klik di sini untuk mendownload dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

## 5. Persiapan

### 5.1 Persiapan Direktori dan Environment

Untuk menyiapkan _virtual environment_ Anda, buka IDE kesayangan Anda. Jika menggunakan Visual Studio Code, Anda langsung buka aplikasinya kemudian buka folder **a590_proyek_pertama**. Misalnya Anda menyimpan folder tersebut di direktori berikut:
`D:\class\data_science\dicoding\a590_proyek_pertama`

Setelah berada di direktori ini, buka terminal di VS Code (bisa menggunakan shortcut **ctrl+shift+`**). Setelah terminal terbuka, ketikkan perintah berikut:

**1. Membuat virtual environment**

```bash
D:\class\data_science\dicoding\a590_proyek_pertama> python -m venv .venv
```

**2. Mengaktifkan environment (Windows)**

```bash
D:\class\data_science\dicoding\a590_proyek_pertama> .venv\Scripts\activate
```

_(Direktori akan berubah menjadi seperti dibawah ini)_

```bash
(.venv) D:\class\data_science\dicoding\a590_proyek_pertama>
```

**3. Mengaktifkan environment (Mac/Linux)**

```bash
$source .venv/bin/activate
```

_(Direktori akan berubah menjadi seperti di bawah ini:)_

```bash
(.venv) $
```

**4. Menginstal semua library yang dibutuhkan sekaligus dari file requirements.txt**

```bash
(.venv) D:\class\data_science\dicoding\a590_proyek_pertama> pip install -r requirements.txt
```

**Berikut adalah isi requirements:**

- anyio==4.13.0
- argon2-cffi==25.1.0
- argon2-cffi-bindings==25.1.0
- arrow==1.4.0
- asttokens==3.0.1
- async-lru==2.3.0
- attrs==26.1.0
- babel==2.18.0
- beautifulsoup4==4.14.3
- bleach==6.3.0
- certifi==2026.2.25
- cffi==2.0.0
- charset-normalizer==3.4.6
- colorama==0.4.6
- comm==0.2.3
- contourpy==1.3.2
- cycler==0.12.1
- debugpy==1.8.20
- decorator==5.2.1
- defusedxml==0.7.1
- exceptiongroup==1.3.1
- executing==2.2.1
- fastjsonschema==2.21.2
- fonttools==4.62.1
- fqdn==1.5.1
- h11==0.16.0
- httpcore==1.0.9
- httpx==0.28.1
- idna==3.11
- ipykernel==7.2.0
- ipython==8.38.0
- isoduration==20.11.0
- jedi==0.19.2
- Jinja2==3.1.6
- joblib==1.5.1
- json5==0.13.0
- jsonpointer==3.1.1
- jsonschema==4.26.0
- jsonschema-specifications==2025.9.1
- jupyter-events==0.12.0
- jupyter-lsp==2.3.0
- jupyter_client==8.8.0
- jupyter_core==5.9.1
- jupyter_server==2.17.0
- jupyter_server_terminals==0.5.4
- jupyterlab==4.5.6
- jupyterlab_pygments==0.3.0
- jupyterlab_server==2.28.0
- kiwisolver==1.5.0
- lark==1.3.1
- MarkupSafe==3.0.3
- matplotlib==3.10.3
- matplotlib-inline==0.1.7
- mistune==3.2.0
- nbclient==0.10.4
- nbconvert==7.17.0
- nbformat==5.10.4
- nest-asyncio==1.6.0
- notebook==7.5.5
- notebook_shim==0.2.4
- numpy==1.26.4
- overrides==7.7.0
- packaging==26.0
- pandas==2.3.2
- pandocfilters==1.5.1
- parso==0.8.6
- pillow==12.1.1
- platformdirs==4.9.4
- prometheus_client==0.24.1
- prompt_toolkit==3.0.52
- psutil==7.2.2
- pure_eval==0.2.3
- pycparser==3.0
- Pygments==2.19.2
- pyparsing==3.3.2
- python-dateutil==2.9.0.post0
- python-json-logger==4.0.0
- pytz==2026.1.post1
- pywinpty==3.0.3
- PyYAML==6.0.3
- pyzmq==27.1.0
- referencing==0.37.0
- requests==2.33.0
- rfc3339-validator==0.1.4
- rfc3986-validator==0.1.1
- rfc3987-syntax==1.1.0
- rpds-py==0.30.0
- scikit-learn==1.7.1
- scipy==1.15.3
- seaborn==0.13.2
- Send2Trash==2.1.0
- six==1.17.0
- soupsieve==2.8.3
- stack-data==0.6.3
- terminado==0.18.1
- threadpoolctl==3.6.0
- tinycss2==1.4.0
- tomli==2.4.1
- tornado==6.5.5
- traitlets==5.14.3
- typing_extensions==4.15.0
- tzdata==2025.3
- uri-template==1.3.0
- urllib3==2.6.3
- wcwidth==0.6.0
- webcolors==25.10.0
- webencodings==0.5.1
- websocket-client==1.9.0

### 5.2 Menjalankan Program Prediksi (Machine Learning)

Tim HR dapat menggunakan model Machine Learning yang telah dilatih untuk memprediksi status seorang karyawan di masa depan. Berikut adalah panduan penggunaan untuk pengguna non-teknis:

- Buka terminal. Misalnya anda menyimpan proyek di D:\class\data_science\dicoding\kelas_mahir\proyek_pertama\a590-Belajar-Penerapan-Data-Science-a590_proyek_pertama\a590_proyek_pertama
- Aktifkan .venv

```bash
D: (enter)
cd D:\class\data_science\dicoding\kelas_mahir\proyek_pertama\a590-Belajar-Penerapan-Data-Science-a590_proyek_pertama\a590_proyek_pertama (enter)
.venv\Scripts\activate (enter)
```

- Panggil prediction dengan cara berikut, kemudian klik enter

```bash
python prediction.py (enter)
```

- Anda akan diminta memasukkan EmployeeId (dalam dunia nyata bisa berupa nama). Misalnya masukkan EmployeeId seperti "1, 2, 10"
- Outputnya akan keluar
- Jika ingin memprediksi kembali dan terminal masih terbuka dengan .venv masih aktif, anda tinggal ketik kembali `python prediction.py`, kemudian masukkan EmployeeId

**Contoh Output Hasil Prediksi**

```bas
======================================================
  Sistem Prediksi Attrition HR - Jaya Jaya Maju
======================================================

Petunjuk: Anda bisa memasukkan lebih dari satu ID dengan memisahkannya menggunakan koma.
Contoh: 1, 2, 45

👉 Masukkan EmployeeId yang ingin diprediksi: 1,150,500,1000000

Memproses prediksi...

------------------------------------------------------
[1] ✅ AMAN       : Karyawan ini diprediksi BERTAHAN.
[150] ✅ AMAN       : Karyawan ini diprediksi BERTAHAN.
❌ Karyawan dengan EmployeeId 500 tidak ditemukan di database.
❌ Karyawan dengan EmployeeId 1000000 tidak ditemukan di database.
------------------------------------------------------

Selesai...
```

### 5.3 Mengakses Business Dashboard

Link Dashboard Tableau: Klik di sini untuk mengakses Dashboard Attrition Jaya Jaya Maju
**Link dashboard:** [Klik di sini untuk mengakses dashboard](https://public.tableau.com/app/profile/zulfi.sam.shiddiq/viz/DashoardAttrition-JayaJayaMaju/Dashboard1?publish=yes)

## 6. Business Dashboard

Dashboard HR dirancang sebagai alat bantu pengambilan keputusan yang komprehensif. Berikut adalah rincian fitur dan visualisasi di dalamnya:

- 6 KPI Utama: Menyajikan ringkasan metrik esensial secara instan (Total Karyawan, Karyawan Resign, Bertahan, Attrition Rate, Rata-rata Usia, dan Rata-rata Gaji).
- Attrition by OverTime: Memperlihatkan secara jelas bahwa tingginya frekuensi lembur berbanding lurus dengan tingginya jumlah karyawan yang resign.
- Attrition by Monthly Income: Menunjukkan distribusi karyawan yang resign, yang mayoritas menumpuk pada kelompok dengan gaji bulanan di bawah $5.000.
- Attrition by Age Group: Mengelompokkan usia karyawan (Age bins) untuk menunjukkan kerentanan turnover pada demografi karyawan muda (usia 20-30an).
- Avg Tenure by Gender & Attrition: Membandingkan rata-rata lama masa kerja berdasarkan gender dan status keberlanjutan mereka di perusahaan.
- Fitur Filter Interaktif: Terdapat filter dinamis di sisi kanan (Department, Job Role, Age, Gender, Over Time) yang terhubung ke seluruh dashboard. Fitur ini memungkinkan pengguna (Manajer HR) untuk membedah data (slice and dice) secara spesifik guna menemukan anomali atau pola di departemen tertentu tanpa harus membaca data mentah.

## 7. Conclusion

### 7.1 Analisis (EDA & Dashboard)

Berdasarkan eksplorasi data yang mendalam, faktor utama yang paling memengaruhi tingginya tingkat attrition di Jaya Jaya Maju adalah:

- **Beban Kerja (Overtime):** Karyawan yang sering lembur mengalami tingkat resign yang jauh lebih tinggi (indikasi burnout atau kurangnya work-life balance).
- **Kompensasi (Monthly Income):** Tingkat attrition menyusut drastis seiring dengan tingginya pendapatan. Karyawan bergaji di bawah $5.000 rentan mencari peluang di tempat lain.
- **Usia & Masa Kerja:** Karyawan berusia muda (20-30 tahun) adalah demografi yang paling sering melakukan attrition, terutama di departemen Research & Development dan Sales. Selain itu, fase paling kritis di mana karyawan sering memutuskan resign adalah pada masa bakti tahun ke-3 hingga ke-5.

### 7.2 Model Machine Learning

Proyek ini berhasil mengimplementasikan model algoritma Random Forest yang mampu mempelajari pola kompleks dari data historis. Hasilnya, perusahaan tidak lagi hanya bertindak reaktif terhadap laporan masa lalu, melainkan dapat menggunakan model ini (`prediction.py`) secara proaktif untuk mendeteksi dini karyawan berisiko tinggi sebelum mereka benar-benar mengajukan surat resign.

### 7.3 Rekomendasi Action Items

- **Evaluasi Kebijakan Lembur:** Lakukan audit beban kerja di departemen R&D dan Sales. Batasi jam lembur yang berlebihan atau berikan kompensasi/hari libur pengganti yang sepadan.
- **Penyesuaian Kompensasi Karyawan Muda:** Tinjau ulang standar gaji pokok untuk karyawan entry-level atau junior agar lebih kompetitif di pasar.
- **Program Retensi Terarah:** Fokuskan program pengembangan karir atau penyesuaian benefit bagi karyawan yang sedang berada di fase jenuh (masa kerja 3 sampai 5 tahun).
- **Tindakan Preventif (Stay Interview):** Gunakan output dari model Machine Learning untuk menjadwalkan sesi Stay Interview (diskusi empat mata) kepada karyawan yang terdeteksi "Beresiko Resign", guna mencari tahu keluhan mereka dan menawarkan solusi retensi sebelum terlambat.
