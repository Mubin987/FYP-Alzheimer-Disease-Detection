from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import requests
import gdown

# Initialize FastAPI app
app = FastAPI()

frontend_url = "http://127.0.0.1:5500"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Path to upload directory and model
UPLOAD_DIR = "./uploads/"
LOCAL_MODEL_PATH = "./model/best_model_20250514_140846.pth"
CACHE_MODEL_PATH = os.path.join(tempfile.gettempdir(), "best_model_20250514_140846.pth")
MODEL_URL = os.getenv("MODEL_URL")

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define class names and mappings
CLASS_NAMES = ["AD", "MCI", "CN"]
CLASS_MAPPINGS = {
    "AD": "Alzheimer's Disease",
    "MCI": "Mild Cognitive Impairment",
    "CN": "Cognitive Normal"
}

# Define the ResNet3D model architecture exactly as in your training code
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18_3D(num_classes=3):
    return ResNet3D(ResidualBlock, [2, 2, 2, 2], num_classes)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def _get_drive_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def _normalize_drive_url(url):
    marker = "/file/d/"
    if marker not in url:
        return url
    file_id = url.split(marker, 1)[1].split("/", 1)[0]
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_model_if_needed():
    min_valid_bytes = 5 * 1024 * 1024
    if os.path.exists(CACHE_MODEL_PATH):
        if os.path.getsize(CACHE_MODEL_PATH) >= min_valid_bytes:
            with open(CACHE_MODEL_PATH, "rb") as f:
                if f.read(1) != b"<":
                    return CACHE_MODEL_PATH
        os.unlink(CACHE_MODEL_PATH)

    if not MODEL_URL:
        return LOCAL_MODEL_PATH

    os.makedirs(os.path.dirname(CACHE_MODEL_PATH), exist_ok=True)

    temp_path = f"{CACHE_MODEL_PATH}.download"
    download_url = _normalize_drive_url(MODEL_URL)
    if "drive.google.com" in download_url:
        gdown.download(download_url, temp_path, quiet=False, fuzzy=True)
    else:
        session = requests.Session()
        response = session.get(download_url, stream=True, timeout=120)
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        response.close()

    with open(temp_path, "rb") as f:
        if f.read(1) == b"<":
            os.unlink(temp_path)
            raise ValueError("Downloaded HTML instead of model file. Check Drive link sharing.")

    if os.path.getsize(temp_path) < min_valid_bytes:
        os.unlink(temp_path)
        raise ValueError("Downloaded file is too small to be a valid model.")

    os.replace(temp_path, CACHE_MODEL_PATH)

    return CACHE_MODEL_PATH

# Try to load the model using the correct architecture
resolved_model_path = None
try:
    resolved_model_path = download_model_if_needed()

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

    print(f"Loading model from: {resolved_model_path}")
    
    # Try first with our custom ResNet18_3D
    model = ResNet18_3D(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(resolved_model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print("Successfully loaded model with custom ResNet18_3D architecture")
    model_loaded = True
except Exception as e1:
    print(f"Error loading model with ResNet18_3D: {e1}")
    try:
        # If torchvision is available, try with r3d_18
        if not resolved_model_path:
            raise RuntimeError("Model path was not resolved.")
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(resolved_model_path, map_location=device, weights_only=False))
        model = model.to(device)
        model.eval()
        print("Successfully loaded model with torchvision's r3d_18 architecture")
        model_loaded = True
    except Exception as e2:
        print(f"Error loading model with r3d_18: {e2}")
        print("Using uninitialized model - predictions will be random")
        model = ResNet18_3D(num_classes=len(CLASS_NAMES)).to(device)
        model.eval()
        model_loaded = False

def preprocess_npy(file_data):
    """
    Preprocess the uploaded NPY file data for the model, matching the training preprocessing
    """
    try:
        # Load the numpy array
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        # Load the file from disk
        volume = np.load(temp_file_path).astype(np.float32)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Print original shape for debugging
        print(f"Original input shape: {volume.shape}")
        
        # Check if the volume is 3D
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        
        # Normalize each volume to zero mean and unit variance (exactly as in training)
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-7)
        
        # Add channel dimension and replicate to 3 channels (for pretrained models)
        volume = np.expand_dims(volume, axis=0)         # shape (1, D, H, W)
        volume = np.repeat(volume, 3, axis=0)           # shape (3, D, H, W)
        
        # Add batch dimension
        volume = np.expand_dims(volume, axis=0)          # shape (1, 3, D, H, W)
        
        print(f"Final preprocessed shape: {volume.shape}")
        return volume.astype(np.float32)
    
    except Exception as e:
        raise ValueError(f"Error preprocessing NPY file: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint to verify the server is running."""
    if model_loaded:
        return {"message": "FastAPI server is running with PyTorch model successfully loaded!"}
    else:
        return {"message": "FastAPI server is running but model could not be loaded properly"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload a .npy file and get a prediction."""
    try:
        if not file.filename.endswith('.npy'):
            return JSONResponse(
                {"error": "Only .npy files are supported"}, 
                status_code=400
            )
        
        # Read the file content
        file_content = await file.read()
        
        # Preprocess the NPY data
        preprocessed_volume = preprocess_npy(file_content)
        
        # Convert to PyTorch tensor and move to device
        input_tensor = torch.from_numpy(preprocessed_volume).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Get probabilities
            probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
            
            # Get predicted class
            predicted_class_idx = outputs.argmax(dim=1).item()
            predicted_class = CLASS_NAMES[predicted_class_idx]
            full_form = CLASS_MAPPINGS.get(predicted_class, "Unknown Result")
            
            # Create confidence scores
            confidence_scores = {CLASS_MAPPINGS[cls]: f"{prob:.2%}" for cls, prob in zip(CLASS_NAMES, probabilities)}
            
            response = {
                "message": full_form,
                "class": predicted_class,
                "confidence": confidence_scores
            }
            
            # Add warning if model was not loaded successfully
            if not model_loaded:
                response["warning"] = "Model was not loaded properly - predictions may not be accurate"
            
            return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# Run the server with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)