const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

const getFacePrediction = async (imagePath) => {
  const formData = new FormData();

  formData.append("file", fs.createReadStream(imagePath));

  const response = await axios.post(
    "http://127.0.0.1:8000/predict",
    formData,
    {
      headers: formData.getHeaders(),
    }
  );

  return response.data;
};

module.exports = {
  getFacePrediction,
};