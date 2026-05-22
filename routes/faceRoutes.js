const express = require("express");
const multer = require("multer");
const fs = require("fs");

const router = express.Router();

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },

  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage });

router.get("/test", (req, res) => {
  res.json({
    message: "API berhasil",
  });
});

router.post("/faces/analyze", upload.single("image"), async (req, res) => {
  try {
    // Validasi ketersedian File
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "Foto belum diupload",
      });
    }

    const imagePath = req.file.path;

    console.log("Foto diterima:", imagePath);

    const result = "Oval";

    const hairstyles = {
      Oval: [
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

      Round: [
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

      Square: [
        "Burr Cut",
        "Butch Cut",
        "Flat Top",
        "Regulation Cut",
        "High and Tight",
        "Crew Cut",
        "Buzz Cut",
        "Short Quiff",
      ],

      Diamond: [
        "Flow Hairstyle",
        "Medium Length Haircut",
        "Slick Back",
        "Tousled Undercut",
        "Side Part",
        "Textured Crop",
        "Messy Fringe",
      ],

      Heart: [
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

      Oblong: [
        "Curtain Hair",
        "Layered Cut",
        "Side Fringe",
        "Textured Crop",
        "Middle Part",
        "Classic Side Part",
        "Brush Up",
      ],
    };

    const hairstyleList = hairstyles[result];

    /* Delete */
    fs.unlinkSync(imagePath);

    res.json({
      success: true,

      faceShape: result,

      hairstyle: hairstyleList,

      message: "Foto berhasil dianalisis",
    });
  } catch (error) {
    console.log("ERROR ASLI:", error);

    res.status(500).json({
      success: false,

      error: error.message,

      message: "Terjadi error",
    });
  }
});

module.exports = router;
