const { getFacePrediction } = require("../services/faceService");

const predictFaceShape = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        message: "Image file is required",
      });
    }

    const result = await getFacePrediction(req.file.path);

    return res.status(200).json({
      message: "Prediction success",
      data: result,
    });
  } catch (error) {
    return res.status(500).json({
      message: "Prediction failed",
      error: error.message,
    });
  }
};

module.exports = {
  predictFaceShape,
};