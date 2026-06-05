// Mengimpor fungsi getFacePrediction dari layer service untuk menangani logika pemanggilan ke ML service
const { getFacePrediction } = require("../services/faceService");

/**
 * Controller untuk menangani request prediksi bentuk wajah berdasarkan file gambar yang diunggah.
 * Menggunakan middleware pencatat file (seperti Multer) untuk menangkap file dari client.
 */
const predictFaceShape = async (req, res) => {
  try {
    // Validasi awal: Memastikan client telah mengirimkan file gambar dalam request
    if (!req.file) {
      return res.status(400).json({
        message: "Image file is required",
      });
    }

    // Memanggil fungsi service dengan mengoper path/lokasi penyimpanan sementara file gambar tersebut
    const result = await getFacePrediction(req.file.path);

    // Mengembalikan respon sukses (HTTP 200) beserta data hasil prediksi dari service
    return res.status(200).json({
      message: "Prediction success",
      data: result,
    });
  } catch (error) {
    // Menangkap error jika terjadi kegagalan sistem atau kegagalan koneksi ke service ML (HTTP 500)
    return res.status(500).json({
      message: "Prediction failed",
      error: error.message,
    });
  }
};

// Mengekspor fungsi controller agar dapat digunakan dan didaftarkan pada berkas router Express
module.exports = {
  predictFaceShape,
};