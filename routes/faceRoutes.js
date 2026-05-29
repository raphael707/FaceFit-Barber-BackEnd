const express = require("express");
const multer = require("multer");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

const router = express.Router();

if (!fs.existsSync("uploads")) {
  fs.mkdirSync("uploads");
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },

  filename: (req, file, cb) => {
    const safeFileName = file.originalname.replace(/\s+/g, "-");
    cb(null, Date.now() + "-" + safeFileName);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

    if (!allowedTypes.includes(file.mimetype)) {
      return cb(new Error("File harus berupa gambar JPG, PNG, atau WEBP"));
    }

    cb(null, true);
  },
});

const predictFaceShape = async (imagePath) => {
  const formData = new FormData();

  formData.append("file", fs.createReadStream(imagePath));

  const response = await axios.post("http://127.0.0.1:8000/predict", formData, {
    headers: formData.getHeaders(),
  });

  return response.data;
};

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

router.get("/test", (req, res) => {
  res.json({
    message: "API berhasil",
  });
});

router.post("/faces/analyze", upload.single("image"), async (req, res) => {
  let imagePath = null;

  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "Foto belum diupload",
      });
    }

    imagePath = req.file.path;

    console.log("Foto diterima:", imagePath);

    const prediction = await predictFaceShape(imagePath);

    const result = prediction.face_shape;
    const hairstyleList = hairstyles[result] || [];

    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }

    return res.json({
      success: true,
      faceShape: result,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      hairstyle: hairstyleList,
      message: "Foto berhasil dianalisis",
    });
  } catch (error) {
    console.log("ERROR ASLI:", error.response?.data || error.message);

    if (imagePath && fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
    }

    return res.status(500).json({
      success: false,
      error: error.response?.data || error.message,
      message: "Terjadi error",
    });
  }
});

router.use((error, req, res, next) => {
  console.log("UPLOAD ERROR:", error.message);

  return res.status(400).json({
    success: false,
    message: "Upload file gagal",
    error: error.message,
  });
});

module.exports = router;