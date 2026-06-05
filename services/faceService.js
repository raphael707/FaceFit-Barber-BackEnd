const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

/**
 * Fungsi service untuk mengirimkan file gambar dari server Node.js ke service Machine Learning (FastAPI).
 * Fungsi ini bertindak sebagai jembatan (proxy/client) antar-service (Microservices communication).
 * * @param {string} imagePath - Path/lokasi lokal tempat gambar disimpan sementara oleh Multer.
 * @returns {Promise<Object>} - Data JSON hasil prediksi bentuk wajah, confidence score, dan probabilitas dari model ML.
 */
const getFacePrediction = async (imagePath) => {
  // Membuat instance FormData baru untuk menyusun payload berbentuk multipart/form-data (seperti form HTML)
  const formData = new FormData();

  // Membaca file gambar dari sistem penyimpanan lokal menggunakan Stream (fs.createReadStream).
  // Menggunakan stream jauh lebih hemat memori (RAM) dibandingkan membaca seluruh file sekaligus ke dalam memori.
  // Data dikirimkan ke form-data dengan kunci (key) bernama "file" sesuai format yang diminta oleh FastAPI.
  formData.append("file", fs.createReadStream(imagePath));

  // Mengirimkan request HTTP POST ke endpoint service ML FastAPI (http://127.0.0.1:8000/predict)
  const response = await axios.post(
    "http://127.0.0.1:8000/predict",
    formData,
    {
      // formData.getHeaders() otomatis menghasilkan header 'content-type' yang tepat,
      // termasuk string 'boundary' unik yang diperlukan server untuk memisahkan part data file.
      headers: formData.getHeaders(),
    }
  );

  // Mengembalikan properti data dari objek response Axios yang berisi payload JSON dari FastAPI
  return response.data;
};

// Mengekspor fungsi service agar dapat diimpor dan digunakan oleh layer controller (seperti faceController.js)
module.exports = {
  getFacePrediction,
};