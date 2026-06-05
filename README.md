# FaceFit Barber BackEnd - API and AI Service for Face Shape Analysis and Hairstyle Recommendation

"FaceFit Barber BackEnd" is the backend service for the FaceFit Barber capstone project. This backend is designed to process user face images, analyze face shape using an AI model, and return personalized hairstyle recommendations based on the detected face shape.

![logoApp](./assets/LogoApp.png)

The backend consists of two main parts: an **Express.js API** for local backend routing and a **FastAPI ML Service** for running the TensorFlow/Keras face shape classification model. The AI service is also deployed using **Hugging Face Spaces** so it can be accessed online by the frontend application.

## Main Features

* 📷 **Image Upload Handling**: Receives face images from the frontend using multipart/form-data.
* 🧠 **AI Face Shape Classification**: Uses a TensorFlow/Keras `.keras` model to classify the user's face shape.
* ✂️ **Hairstyle Recommendation**: Returns hairstyle suggestions based on the predicted face shape.
* 🚀 **FastAPI ML Service**: Provides a dedicated API service for AI model inference.
* 🌐 **Hugging Face Deployment**: AI backend service is deployed online using Hugging Face Spaces.
* 🔗 **Frontend Integration**: Provides API endpoints that can be connected to the React/Vite frontend.

## Technologies Used

* Node.js
* Express.js
* FastAPI
* Python
* TensorFlow/Keras
* NumPy
* Pillow
* Multer
* Axios
* Hugging Face Spaces

## Project Structure

```text
FaceFit-Barber-BackEnd/
├── controllers/
│   └── faceController.js
├── middlewares/
│   └── uploadMiddleware.js
├── ml-service/
│   ├── model/
│   │   └── face_shape_model.keras
│   ├── custom_layers.py
│   ├── main.py
│   └── requirements.txt
├── routes/
│   └── faceRoutes.js
├── services/
│   └── faceService.js
├── uploads/
├── .gitignore
├── app.js
├── package.json
├── package-lock.json
└── README.md
```

## AI Model Overview

The AI model is used to classify the user's face shape from an uploaded image. The model was built using **TensorFlow/Keras** and saved in `.keras` format.

The model receives image input with the following shape:

```text
224 x 224 x 3
```

The model predicts one of the following face shape classes:

```text
diamond
heart
oval
round
square
```

Before prediction, the image is processed through several preprocessing steps:

1. Convert image to RGB format.
2. Apply center crop with a ratio of 0.88.
3. Resize image to 224 x 224 pixels.
4. Convert image to `float32` array.
5. Send the processed image to the AI model.

The backend does not manually divide the image array by `/255`, because the model already includes a `Rescaling` layer inside the architecture.

## API Endpoints

### Test API

```http
GET /api/test
```

Example response:

```json
{
  "message": "API berhasil"
}
```

### Analyze Face Shape

```http
POST /api/faces/analyze
```

Request body uses `multipart/form-data`.

| Key   | Type | Description     |
| ----- | ---- | --------------- |
| image | File | User face image |

Example response:

```json
{
  "success": true,
  "faceShape": "Heart",
  "confidence": 0.637,
  "hairstyle": [
    "Textured Fringe",
    "Side Part",
    "Low Fade"
  ],
  "probabilities": {
    "diamond": 0.257,
    "heart": 0.637,
    "oval": 0.025,
    "round": 0.080,
    "square": 0.000
  },
  "message": "Foto berhasil dianalisis"
}
```

## ML Service Endpoints

The FastAPI ML service provides endpoints for testing and model inference.

```http
GET /
POST /predict
POST /api/faces/analyze
```

The `/predict` endpoint is mainly used for direct model testing and debugging.

The `/api/faces/analyze` endpoint is used by the frontend because it returns the face shape result, confidence score, probabilities, and hairstyle recommendations.

## Prerequisites

* Node.js v18.x or higher
* npm
* Python 3.10 or higher
* pip
* TensorFlow/Keras
* FastAPI
* Uvicorn

## Setup and Installation

### 1. Clone Repository

```bash
git clone https://github.com/raphael707/FaceFit-Barber-BackEnd.git
cd FaceFit-Barber-BackEnd
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Install Python Dependencies

Go to the ML service folder:

```bash
cd ml-service
pip install -r requirements.txt
```

## Running Locally

### 1. Run FastAPI ML Service

Open a terminal and run:

```bash
cd ml-service
uvicorn main:app --reload --port 8000
```

The ML service will run at:

```text
http://127.0.0.1:8000
```

FastAPI documentation can be accessed at:

```text
http://127.0.0.1:8000/docs
```

### 2. Run Express Backend

Open another terminal and run:

```bash
node app.js
```

The Express backend will run at:

```text
http://localhost:3000
```

## Deployment

For deployment, the AI backend service is deployed using **Hugging Face Spaces**. The deployed FastAPI service allows the frontend to send an image and receive the AI prediction result online.

Production AI Backend:

```text
https://justblaisee-facefit-ml-service.hf.space
```

Main endpoint used by the frontend:

```text
https://justblaisee-facefit-ml-service.hf.space/api/faces/analyze
```

## Notes

* Do not commit `node_modules/`.
* Do not commit `.env`.
* Do not commit temporary uploaded images.
* The `.keras` model file is used for AI inference.
* If the model file is too large, use Git LFS.
* Hugging Face Spaces free hardware may sleep when inactive, so the first request after inactivity may take longer.

## Contributing

Feel free to fork this repository and submit a pull request.

## License

MIT License - Use freely with attribution.

## Credits

Developed as part of a **Capstone Project** to improve the barbershop and grooming experience through AI-based face shape analysis and personalized hairstyle recommendation.
