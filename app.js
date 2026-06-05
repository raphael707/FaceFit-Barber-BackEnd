const express = require("express");
const cors = require("cors");

// Mengimpor modul rute (router) yang menangani endpoint analisis wajah
const faceRoutes = require("./routes/faceRoutes");

// Inisialisasi aplikasi Express server
const app = express();

// Mengaktifkan Middleware CORS (Cross-Origin Resource Sharing)
// Mengizinkan aplikasi frontend (seperti React, Vue, atau Mobile App) untuk mengakses API dari domain/port berbeda
app.use(cors());

// Mengaktifkan built-in middleware Express untuk membaca incoming request berupa JSON (JSON body parser)
// Memungkinkan server membaca data JSON yang dikirim client via 'req.body'
app.use(express.json());

// Mendaftarkan modul faceRoutes ke dalam path dasar '/api'
// Semua rute di dalam faceRoutes akan memiliki prefix /api (contoh: /api/faces/analyze)
app.use("/api", faceRoutes);

// Endpoint Root (GET /) sebagai pangkalan utama untuk mengecek status kesehatan server (Health Check)
app.get("/", (req, res) => {
  res.send("Backend FaceFit Barber jalan");
});

// Menentukan port jaringan tempat server Express akan berjalan (Port 3000)
const PORT = 3000;

// Menjalankan (mengaktifkan) server Express agar mulai mendengarkan incoming requests pada port yang ditentukan
app.listen(PORT, () => {
  // Mencetak log ke terminal untuk memastikan server berhasil berjalan dengan sukses
  console.log(`Server running on port ${PORT}`);
});