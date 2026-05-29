from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from custom_layers import (
    SpatialAttentionLayer,
    TemperatureSoftmax,
    FocalCrossEntropyLoss
)

app = FastAPI()

model = tf.keras.models.load_model(
    "model/face_shape_model.keras",
    custom_objects={
        "SpatialAttentionLayer": SpatialAttentionLayer,
        "TemperatureSoftmax": TemperatureSoftmax,
        "FocalCrossEntropyLoss": FocalCrossEntropyLoss,
    },
    compile=False
)

CLASS_NAMES = ["diamond", "heart", "oval", "round", "square"]

@app.get("/")
def root():
    return {
        "message": "ML service FaceFit Barber jalan"
    }

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image).astype("float32")
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        input_data = preprocess_image(image_bytes)
        prediction = model.predict(input_data)

        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        return {
            "success": True,
            "face_shape": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(prediction[0][i])
                for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }