from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Mengimpor layer, komponen loss, dan fungsi aktivasi kustom yang digunakan oleh model TensorFlow
from custom_layers import (
    SpatialAttentionLayer,
    TemperatureSoftmax,
    FocalCrossEntropyLoss
)


# Inisialisasi aplikasi FastAPI
app = FastAPI()


# Konfigurasi Middleware CORS (Cross-Origin Resource Sharing)
# Mengizinkan akses dari semua origin (*), metode HTTP, dan headers untuk integrasi frontend-backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Memuat model TensorFlow/Keras (.keras) yang sudah dilatih
# custom_objects digunakan untuk mendaftarkan layer/loss kustom agar dikenali saat proses loading
# compile=False digunakan karena model di backend ini hanya berfungsi untuk inferensi (prediksi), bukan training
model = tf.keras.models.load_model(
    "model/face_shape_model.keras",
    custom_objects={
        "SpatialAttentionLayer": SpatialAttentionLayer,
        "TemperatureSoftmax": TemperatureSoftmax,
        "FocalCrossEntropyLoss": FocalCrossEntropyLoss,
    },
    compile=False
)


# Daftar label kelas bentuk wajah yang sesuai dengan urutan output (indeks) dari model ML
CLASS_NAMES = [
    "diamond",
    "heart",
    "oval",
    "round",
    "square"
]


# Mapping/pemetaan rekomendasi gaya rambut pria berdasarkan masing-masing bentuk wajah
HAIRSTYLES = {
    "diamond": [
        "Flow Hairstyle",
        "Medium Length Haircut",
        "Slick Back",
        "Tousled Undercut",
        "Side Part",
        "Textured Crop",
        "Messy Fringe",
    ],

    "heart": [
        "Textured Fringe",
        "Side Part",
        "Low Fade",
        "Messy Crop",
        "Curtain Hair",
        "Side Swept Fringe",
        "Messy Quiff",
        "Layered Hair",
        "Textured Crop",
    ],

    "oval": [
        "Buzz Cut",
        "Crew Cut",
        "Caesar Cut",
        "French Crop",
        "Edgar Haircut",
        "Curtain",
        "Two Block",
        "Fluffy Hair",
        "Comma Hair",
        "Slick Back",
        "Textured Quiff",
    ],

    "round": [
        "Blowout",
        "Faux Hawk",
        "High Fade",
        "Ivy League",
        "Pompadour Modern",
        "Quiff",
        "Spiky Hair",
        "Side Part",
        "Textured Top",
    ],

    "square": [
        "Burr Cut",
        "Butch Cut",
        "Flat Top",
        "Regulation Cut",
        "High and Tight",
        "Crew Cut",
        "Buzz Cut",
        "Short Quiff",
    ],
}


# Endpoint Root (GET /) untuk mengecek apakah service API berjalan dengan baik (Health Check)
@app.get("/")
def root():
    return {
        "message": "ML service FaceFit Barber jalan"
    }


# Fungsi penunjang untuk memotong bagian tengah gambar (center crop) agar rasio menjadi 1:1 (kotak)
def center_crop_pil(image, crop_ratio=0.88):
    width, height = image.size

    # Batasi nilai crop_ratio agar tetap berada di rentang aman 0.50 hingga 1.00
    crop_ratio = float(np.clip(crop_ratio, 0.50, 1.00))

    # Tentukan ukuran panjang sisi kotak berdasarkan dimensi terkecil gambar dikali rasio
    side = int(min(width, height) * crop_ratio)
    side = max(side, 1) # Memastikan ukuran sisi minimal 1 piksel

    # Hitung koordinat pembatas (bounding box) untuk pemotongan di tengah koordinat gambar
    left = max(0, (width - side) // 2)
    top = max(0, (height - side) // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)

    # Potong gambar berdasarkan koordinat yang telah dihitung dan kembalikan hasilnya
    return image.crop((left, top, right, bottom))


# Fungsi memproses file mentah (bytes) menjadi array yang siap diumpankan ke model TensorFlow
def preprocess_image(image_bytes):
    # Membuka file gambar dari memory stream dan mengubah modenya ke RGB (3 channel warna)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # CENTER_CROP_RATIO = 0.88
    image = center_crop_pil(image, crop_ratio=0.88)

    # Sesuai input model: 224 x 224 x 3
    # Mengubah ukuran gambar (resize) menjadi 224x224 piksel dengan interpolasi Bilinear
    image = image.resize((224, 224), Image.BILINEAR)

    img_array = np.array(image).astype("float32")

    # Menambahkan dimensi batch (batch dimension) di awal array karena model menerima input berbentuk batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Fungsi utama untuk melakukan prediksi bentuk wajah menggunakan model TensorFlow
def predict_face_shape(image_bytes):
    # Lakukan prapemrosesan gambar terlebih dahulu
    input_data = preprocess_image(image_bytes)

    # Eksekusi prediksi menggunakan model ML
    prediction = model.predict(input_data)

    # Mengambil indeks dengan nilai probabilitas tertinggi dari hasil prediksi batch pertama ([0])
    predicted_index = int(np.argmax(prediction[0]))
    # Mengambil nilai probabilitas tertinggi sebagai tingkat kepercayaan (confidence score)
    confidence = float(np.max(prediction[0]))
    # Mendapatkan string nama bentuk wajah berdasarkan indeks prediksinya
    face_shape = CLASS_NAMES[predicted_index]

    # Menyusun dictionary berisi semua probabilitas untuk tiap kelas agar bisa dianalisis detail di frontend
    probabilities = {
        CLASS_NAMES[i]: float(prediction[0][i])
        for i in range(len(CLASS_NAMES))
    }

    return face_shape, confidence, probabilities


# Endpoint Prediksi Lama/Umum (POST /predict) - Menerima file gambar via form-data
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Membaca data biner (bytes) dari file yang diunggah secara asynchronous
        image_bytes = await file.read()

        # Memanggil fungsi prediksi bentuk wajah
        face_shape, confidence, probabilities = predict_face_shape(image_bytes)

        # Mengembalikan response JSON standar hasil prediksi
        return {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "probabilities": probabilities
        }

    except Exception as e:
        # Menangkap error jika proses pemrosesan/inferensi gagal
        return {
            "success": False,
            "error": str(e)
        }


# Endpoint Analisis Khusus Aplikasi FaceFit Barber (POST /api/faces/analyze)
# Menyediakan data bentuk wajah terkapitalisasi dan langsung menyertakan list rekomendasi gaya rambut
@app.post("/api/faces/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        # Membaca data biner (bytes) dari parameter file bernama 'image'
        image_bytes = await image.read()

        # Memanggil fungsi prediksi bentuk wajah
        face_shape, confidence, probabilities = predict_face_shape(image_bytes)

        # Mengembalikan response JSON lengkap yang diformat khusus untuk kebutuhan UI frontend
        return {
            "success": True,
            "faceShape": face_shape.capitalize(), # Mengubah huruf pertama menjadi kapital (cth: "Oval")
            "confidence": confidence,
            "hairstyle": HAIRSTYLES.get(face_shape, []), # Mengambil list gaya rambut yang cocok
            "probabilities": probabilities,
            "message": "Foto berhasil dianalisis"
        }

    except Exception as e:
        # Menangkap error dan mengembalikan pesan ramah pengguna (user-friendly message)
        return {
            "success": False,
            "error": str(e),
            "message": "Terjadi error saat analisis foto"
        }