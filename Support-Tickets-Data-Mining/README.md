# Project Data Mining: Support Tickets

Project ini melakukan pemahaman data, preprocessing, visualisasi EDA, dan
klasifikasi menggunakan Decision Tree. Skrip membaca CSV lokal terlebih dahulu;
nama dan tipe kolom dideteksi dari isi file, bukan dibuat sebagai data dummy.

## Tujuan dan rumusan masalah

Tujuan project adalah mengeksplorasi dataset tiket bantuan dan, jika kolom label
yang sesuai tersedia, membangun model klasifikasi untuk memperkirakan prioritas
tiket atau status tiket.

Rumusan masalah:

1. Bagaimana struktur, tipe, distribusi, missing values, dan duplikasi data
   support tickets?
2. Seberapa baik Decision Tree mengklasifikasikan target prioritas atau status
   menggunakan fitur yang tidak mengandung identifier atau informasi
   pascakejadian?

Target dipilih secara otomatis setelah CSV dibaca. Urutan prioritas pencarian:
`ticket_priority`, `priority`, `ticket_status`, lalu `status` (pencocokan nama
tidak membedakan huruf besar/kecil dan tanda baca). Jika tidak ada kolom yang
cocok, program tetap melakukan eksplorasi dan menyimpan grafik, lalu
menjelaskan bahwa klasifikasi dilewati.

## Dataset

File project adalah `Support Tickets.csv`. Dataset yang tersedia saat project
ini disusun berisi **4.000 baris dan 10 kolom**:

| Kolom | Tipe saat dibaca | Keterangan |
| --- | --- | --- |
| `ticket_id` | Teks | Identitas tiket; tidak digunakan sebagai fitur model. |
| `opened_at` | Teks tanggal/waktu | Waktu tiket dibuka; diekstrak menjadi tahun, bulan, hari, dan jam. |
| `channel` | Teks | Kanal tiket. |
| `queue` | Teks | Antrian/jenis penanganan. |
| `priority` | Teks | Target klasifikasi yang dipilih bila tersedia. |
| `subject` | Teks | Subjek tiket. |
| `first_response_min` | Numerik | Waktu respons pertama dalam menit; dikeluarkan dari fitur untuk mengurangi kebocoran target. |
| `resolved_at` | Teks tanggal/waktu | Waktu resolusi; dikeluarkan karena terjadi setelah tiket dibuat. |
| `reopened` | Boolean | Status tiket dibuka kembali; dikeluarkan sebagai kemungkinan informasi pascakejadian. |
| `csat` | Numerik | Nilai kepuasan; dikeluarkan karena dapat diketahui setelah penanganan. |

Program memeriksa kolom dari file aktual saat dijalankan. Dataset lain dapat
memiliki susunan atau tipe berbeda. Kolom kategorikal, numerik, dan tanggal
ditentukan berdasarkan data aktual. Baris duplikat penuh dihapus sebelum
pemodelan. Missing pada fitur ditangani di dalam pipeline: kategori dengan
`Tidak diketahui` dan numerik dengan median yang dihitung dari data training.

## Library yang diperlukan

- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Instalasi

Python 3.10 atau lebih baru direkomendasikan. Buka terminal pada direktori
project, lalu jalankan:

```bash
python -m venv .venv
```

Aktifkan virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Pasang semua library:

```bash
python -m pip install -r requirements.txt
```

## Struktur folder

```text
Support-Tickets-Data-Mining/
├── Support Tickets.csv
├── main.py
├── README.md
├── requirements.txt
└── output/
```

Folder `output/` dibuat otomatis oleh program jika belum tersedia.

## Menjalankan program

Pastikan `Support Tickets.csv` berada satu folder dengan `main.py`. Jalankan:

```bash
python main.py
```

Program dapat dijalankan melalui terminal VS Code. Seluruh hasil disimpan ke
folder `output/`.

## Tahapan Data Mining

1. **Data loading** — membaca CSV dengan pandas dan menampilkan ukuran, nama
   kolom, serta lima baris pertama.
2. **Data understanding** — menampilkan tipe data, statistik deskriptif,
   kategorisasi tipe kolom, missing values, duplikat, dan distribusi kategori.
3. **Preprocessing** — menghapus baris duplikat penuh; missing kategori diisi
   dengan label pengganti dan missing numerik dengan median. Imputer dan
   encoder dipasang pada pipeline agar statistik preprocessing dipelajari dari
   data training saja.
4. **EDA** — membuat grafik distribusi target, kategori, dan fitur numerik
   menggunakan Seaborn/Matplotlib. Grafik disimpan dalam folder `output/`.
5. **Pencegahan leakage** — identifier dikeluarkan dari fitur. Kolom yang
   namanya menunjukkan penyelesaian, kepuasan, respons, atau status dibuka
   kembali juga dikecualikan. Kolom tanggal yang menunjukkan waktu tiket
   dibuka diubah menjadi fitur kalender.
6. **Encoding** — `OneHotEncoder` mengubah kategori menjadi kolom numerik;
   `handle_unknown="ignore"` menangani kategori baru yang muncul pada data uji.
7. **Pemodelan** — Decision Tree dilatih dan diuji menggunakan pembagian
   stratified 80:20 dengan random state tetap. Jika target tidak ditemukan,
   hanya satu kelas, atau data per kelas tidak mencukupi, model dilewati dengan
   pesan penjelasan.
8. **Evaluasi** — menampilkan accuracy, precision, recall, F1-score macro,
   classification report per kelas, dan confusion matrix.

## Algoritma Decision Tree

Decision Tree membagi data ke dalam aturan-aturan berdasarkan fitur sehingga
prediksi dapat dibaca sebagai rangkaian keputusan. Algoritma ini mendukung
klasifikasi multi-kelas. Kedalaman pohon dibatasi (`max_depth=6`) untuk
mengurangi kompleksitas; `class_weight="balanced"` digunakan untuk
mempertimbangkan ketidakseimbangan kelas.

## Membaca hasil evaluasi

- **Accuracy**: bagian dari seluruh prediksi yang benar.
- **Precision**: dari prediksi untuk suatu kelas, bagian yang benar.
- **Recall**: dari data aktual suatu kelas, bagian yang berhasil ditemukan.
- **F1-score**: gabungan precision dan recall.
- **Macro average**: menghitung metrik tiap kelas lalu merata-ratakannya dengan
  bobot kelas sama, sehingga kelas minoritas ikut diperhatikan.
- **Confusion matrix**: baris adalah kelas aktual, kolom adalah kelas prediksi.
  Nilai diagonal adalah prediksi benar; nilai di luar diagonal menunjukkan
  kelas yang tertukar.

Accuracy saja tidak cukup, terutama jika distribusi kelas tidak seimbang.
Pertimbangkan precision, recall, F1 per kelas, dan confusion matrix bersama.
Nilai metrik yang tercetak berasal dari eksekusi dataset lokal, bukan hasil
yang dijamin sama untuk dataset lain.

## File hasil

Program menghasilkan, bila tahapan terkait berjalan:

- `output/missing_values.csv` — jumlah dan persentase missing per kolom.
- `output/distribusi_kategorikal.csv` — frekuensi nilai kategori.
- `output/distribusi_target.png` — grafik distribusi target.
- `output/distribusi_kategorikal_*.png` dan `output/distribusi_numerik_*.png` — grafik EDA.
- `output/laporan_evaluasi.txt` — metrik dan laporan klasifikasi.
- `output/confusion_matrix.csv` dan `output/confusion_matrix.png`.
- `output/hasil_prediksi.csv` — label aktual dan prediksi pada data testing.
- `output/ringkasan_analisis.csv` — ringkasan data dan metrik.
- `output/status_model.txt` — status pelatihan atau alasan model dilewati.

## Kesimpulan

Program tidak menetapkan kesimpulan atau klaim performa sebelum dijalankan.
Kesimpulan harus ditulis berdasarkan laporan evaluasi dan distribusi kelas yang
dihasilkan pada folder `output/`, serta mempertimbangkan metrik per kelas dan
confusion matrix.
