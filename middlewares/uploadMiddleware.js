const multer = require("multer");
const path = require("path");
const fs = require("fs");

// Menentukan nama folder tempat penyimpanan sementara file gambar yang diunggah
const uploadDir = "uploads";

// Memeriksa apakah folder penyimpanan sudah ada atau belum
// Jika belum ada, folder akan dibuat secara otomatis secara synchronous saat aplikasi dijalankan
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Konfigurasi penyimpanan diskStorage untuk mengatur lokasi dan nama file secara spesifik
const storage = multer.diskStorage({
  // Menentukan folder tujuan penyimpanan file yang diunggah
  destination: function (req, file, cb) {
    cb(null, uploadDir); // Parameter pertama null untuk error, parameter kedua adalah folder tujuan
  },

  // Menentukan format penamaan file agar unik dan tidak saling menimpa (overwrite)
  filename: function (req, file, cb) {
    // Membuat nama unik menggunakan timestamp saat ini (Date.now()) ditambahkan ekstensi asli file (.jpg, .png, dll)
    const uniqueName = Date.now() + path.extname(file.originalname);
    cb(null, uniqueName);
  },
});

// Menginisialisasi middleware Multer menggunakan konfigurasi storage yang telah dibuat di atas
const upload = multer({ storage });

// Mengekspor middleware upload agar dapat dipasang pada route Express untuk menangani unggahan file
module.exports = upload;