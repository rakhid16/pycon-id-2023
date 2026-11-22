# PyCon ID 2023

Contoh API berbasis FastAPI untuk prediksi spesies bunga Iris dan klasifikasi
gambar kucing atau anjing. Notebook `notebook.ipynb` berisi alur pelatihan
model.

## Persiapan

Gunakan Python 3.11 atau versi yang kompatibel dengan dependensi PyTorch.
Buat dan aktifkan virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Pasang dependensi API:

```bash
python -m pip install fastapi "uvicorn[standard]" joblib numpy scikit-image scikit-learn
```

Untuk menjalankan notebook, pasang juga `pandas`, `matplotlib`, dan Jupyter.
Pasang `torch` dan `torchvision` yang kompatibel dengan sistem operasi dan
perangkat keras Anda. Sebelum menjalankan API, pastikan berkas model berikut
tersedia relatif terhadap direktori proyek:

- `models/iris.pkl`
- `models/image.pkl`

## Menjalankan API

Dari direktori proyek, jalankan:

```bash
uvicorn main:app --reload
```

Dokumentasi interaktif tersedia di <http://127.0.0.1:8000/docs>.

## Endpoint

| Metode | Path | Keterangan |
| --- | --- | --- |
| `GET` | `/` | Pemeriksaan sederhana bahwa API aktif. |
| `GET` | `/items/{item_id}` | Mengembalikan ID item dan parameter opsional `name`. |
| `POST` | `/new_items` | Menerima objek JSON berisi `item_id` dan `name` opsional. |
| `POST` | `/iris` | Memprediksi spesies dari empat ukuran bunga. |
| `POST` | `/cat_dog` | Mengklasifikasikan gambar yang diunggah sebagai kucing atau anjing. |

Contoh prediksi Iris:

```bash
curl -X POST http://127.0.0.1:8000/iris \
  -H "Content-Type: application/json" \
  -d '{
    "panjang_kelopak": 5.1,
    "lebar_kelopak": 3.5,
    "panjang_mahkota": 1.4,
    "lebar_mahkota": 0.2
  }'
```

Contoh klasifikasi gambar:

```bash
curl -X POST http://127.0.0.1:8000/cat_dog \
  -F "file=@gambar.jpg"
```

## Pelatihan model

Notebook menggunakan `iris.csv` untuk model Iris. Untuk model gambar, siapkan
dataset dalam struktur folder yang dibaca `ImageFolder`:

```text
data/
├── latih/
│   ├── cat/
│   └── dog/
└── uji/
    ├── cat/
    └── dog/
```

Jalankan sel-sel notebook untuk melatih dan menyimpan model ke direktori
`models/`.
