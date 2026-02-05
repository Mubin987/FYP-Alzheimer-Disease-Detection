from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from utils import preprocess_mri

# Initialize FastAPI app
app = FastAPI()

frontend_url = "http://127.0.0.1:5500"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],  # Add specific origins
    allow_credentials=False,       # Disable credentials unless required
    allow_methods=["GET", "POST"], # Allow only required methods
    allow_headers=["*"],           # Allow all headers (adjust if needed)
)

# Path to upload directory and model
UPLOAD_DIR = "./uploads/"
MODEL_PATH = "./model/best_model_20250514_140846.pth"

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint to verify the server is running."""
    return {"message": "FastAPI server is running!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload a .nii file and get a prediction."""
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Preprocess the file
        preprocessed_image = preprocess_mri(file_path)

        # Add batch dimension for prediction
        # preprocessed_image = preprocessed_image[np.newaxis, ..., np.newaxis]

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction,axis=1)[0]
        classes = ["AD", "MCI", "CN"]
        result = classes[predicted_class]
        result_mapping = {
            "AD": "Alzheimer's Disease",
            "MCI": "Mild Cognitive Impairment",
            "CN": "Cognitive Normal"
        }

        full_form = result_mapping.get(result, "Unknown Result")
        #print(f"Prediction: {full_form}")
        return {"message": {full_form}}
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
