const express = require("express");
const multer = require("multer");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

// Inisialisasi router Express untuk memisahkan rute API ke modul tersendiri
const router = express.Router();

// Memastikan folder "uploads" sudah ada di server lokal sebelum menerima file gambar
// Jika folder belum ada, maka folder baru akan dibuat secara otomatis
if (!fs.existsSync("uploads")) {
  fs.mkdirSync("uploads");
}

// Konfigurasi penyimpanan diskStorage milik Multer
const storage = multer.diskStorage({
  // Menentukan folder tujuan penyimpanan file gambar yang diunggah
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },

  // Mengatur nama file agar aman dan unik
  filename: (req, file, cb) => {
    // Mengganti spasi pada nama file asli dengan karakter tanda hubung (-) agar ramah URL
    const safeFileName = file.originalname.replace(/\s+/g, "-");
    // Menggabungkan timestamp unik (Date.now()) dengan nama file yang sudah dibersihkan
    cb(null, Date.now() + "-" + safeFileName);
  },
});

// Menginisialisasi middleware Multer beserta validasi ukuran dan format file
const upload = multer({
  storage,
  limits: {
    fileSize: 5 * 1024 * 1024, // Batasi ukuran maksimal file gambar: 5 Megabytes (MB)
  },
  fileFilter: (req, file, cb) => {
    // Daftar tipe MIME (MIME-types) gambar yang diizinkan oleh aplikasi
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

    // Jika tipe file gambar yang diunggah tidak terdaftar di allowedTypes
    if (!allowedTypes.includes(file.mimetype)) {
      return cb(new Error("File harus berupa gambar JPG, PNG, atau WEBP"));
    }

    // Izinkan proses upload jika file valid
    cb(null, true);
  },
});

// Fungsi penunjang untuk meneruskan file gambar dari server Express ke service ML (FastAPI) via HTTP POST
const predictFaceShape = async (imagePath) => {
  const formData = new FormData();

  // Membaca file gambar dari local path secara streaming lalu memasukkannya ke form-data dengan key "file"
  formData.append("file", fs.createReadStream(imagePath));

  // Mengirim data gambar ke service ML (FastAPI yang berjalan di port 8000 pada endpoint /predict)
  const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
    // getHeaders() diperlukan agar Axios mengirim header Multipart/Form-Data beserta boundary-nya yang benar
    headers: formData.getHeaders(),
  });

  // Mengembalikan data JSON hasil respons dari service ML
  return response.data;
};

// Objek mapping daftar rekomendasi gaya rambut pria berdasarkan bentuk wajah yang terdeteksi
const hairstyles = {
  diamond: [
    "Flow Hairstyle",
    "Medium Length Haircut",
    "Slick Back",
    "Tousled Undercut",
    "Side Part",
    "Textured Crop",
    "Messy Fringe",
  ],

  heart: [
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

  oval: [
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

  round: [
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

  square: [
    "Burr Cut",
    "Butch Cut",
    "Flat Top",
    "Regulation Cut",
    "High and Tight",
    "Crew Cut",
    "Buzz Cut",
    "Short Quiff",
  ],
};

// Endpoint GET /test untuk memeriksa apakah routing API Express berfungsi dengan normal (Ping test)
router.get("/test", (req, res) => {
  res.json({
    message: "API berhasil",
  });
});

// Endpoint Utama POST /faces/analyze untuk memproses analisis bentuk wajah dan memberikan rekomendasi gaya rambut
// upload.single("image") menandakan endpoint ini menerima satu file kiriman form-data dengan key "image"
router.post("/faces/analyze", upload.single("image"), async (req, res) => {
  let imagePath = null;

  try {
    // Validasi: Cek apakah middleware Multer berhasil menangkap kiriman file gambar
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "Foto belum diupload",
      });
    }

    // Menyimpan local path file gambar yang telah disimpan Multer ke folder 'uploads/'
    imagePath = req.file.path;

    console.log("Foto diterima:", imagePath);

    // Mengirim gambar tersebut ke fungsi penunjang predictFaceShape untuk diproses model ML
    const prediction = await predictFaceShape(imagePath);

    // Mengambil string label bentuk wajah hasil prediksi (cth: "oval", "round", dll)
    const result = prediction.face_shape;
    // Mengambil daftar list gaya rambut yang sesuai dari objek mapping hairstyles (default array kosong jika tidak ditemukan)
    const hairstyleList = hairstyles[result] || [];

    // Penghapusan file (Cleanup): Menghapus file gambar sementara di local storage agar server tidak penuh
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }

    // Mengembalikan respon JSON terstruktur yang komplit kembali ke client/frontend
    return res.json({
      success: true,
      faceShape: result,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      hairstyle: hairstyleList,
      message: "Foto berhasil dianalisis",
    });
  } catch (error) {
    // Mencetak log detail error ke terminal server guna mempermudah proses debugging backend
    console.log("ERROR ASLI:", error.response?.data || error.message);

    // Penanganan Cleanup di blok catch: Jika proses prediksi gagal di tengah jalan, pastikan file gambar tetap dihapus
    if (imagePath && fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }

    // Mengembalikan respon error internal server (HTTP 500) ke client
    return res.status(500).json({
      success: false,
      error: error.response?.data || error.message,
      message: "Terjadi error",
    });
  }
});

// Middleware Error Handling khusus Express untuk menangkap error yang dipicu oleh Multer
// Contoh: Ketika user mengunggah file yang ukurannya melebihi batasan limit atau format tidak sesuai kriteria fileFilter
router.use((error, req, res, next) => {
  console.log("UPLOAD ERROR:", error.message);

  return res.status(400).json({
    success: false,
    message: "Upload file gagal",
    error: error.message,
  });
});

// Mengekspor objek router agar dapat dipasang pada file utama aplikasi server (seperti index.js atau app.js)
module.exports = router;