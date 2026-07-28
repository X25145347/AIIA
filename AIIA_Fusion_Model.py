import torch
from joblib import load
import numpy as np
import torch.nn as nn
from torchvision import models, transforms 
from PIL import Image
import os
import jpegio as jio
import png
import pandas as pd

def extract_metadata_features(img, image_path):
    ext = os.path.splitext(image_path)[1].lower()
    img = Image.open(image_path)

    width, height = img.size
    file_size = os.path.getsize(image_path)
    bytes_per_pixel = file_size / (width * height)

    metadata = {
        "width": width,
        "height": height,
        "bytes_per_pixel": bytes_per_pixel,
        "mode": img.mode,
        "icc_present": "icc_profile" in img.info,
    }
    print(type("icc_profile" in img.info))
    # JPEG metadata
    if ext in [".jpg", ".jpeg"]:
        jpeg = jio.read(image_path)
        qtables = jpeg.quant_tables
        metadata["num_qtables"] = len(qtables)
        print("Test")
        if len(qtables) > 0:
            metadata["has_iCCP"]= 0
            metadata["has_tEXt"]= 0
            metadata["num_chunks"]= 0
            metadata["qt_mean_0"] = qtables[0].mean() if len(qtables) > 0 else 0
            metadata["qt_std_0"] = qtables[0].std() if len(qtables) > 0 else 0

    # PNG metadata
    if ext == ".png":
        reader = png.Reader(filename=image_path)
        chunks = list(reader.chunks())
        chunk_types = [ct.decode("ascii") if isinstance(ct, bytes) else ct for ct, _ in chunks]
        metadata["num_chunks"] = len(chunks)
        metadata["has_iCCP"] = int("iCCP" in chunk_types)
        metadata["has_tEXt"] = int("tEXt" in chunk_types)
        metadata["qt_std_0"] = 0
        metadata["qt_mean_0"] = 0
        metadata["num_qtables"] = 0

    return metadata

class EfficientNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.efficientnet_b0(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("./20260728_221253.jpg")
image_tensor = preprocess(img).unsqueeze(0)   # add batch dimension

state_dict = torch.load("efficientnet_ai_vs_real.pth", map_location="cpu")
pixel_model = EfficientNetModel()
pixel_model.load_state_dict(state_dict)
pixel_model.eval()

# Load metadata model (RandomForest)
meta_model = load("metadata_forensic_model.joblib")

with torch.no_grad():
    outputs = pixel_model(image_tensor)
    pixel_prob = torch.softmax(outputs, dim=1)[0,1].item()

meta_features_dict = extract_metadata_features(img, "./20260728_221253.jpg")

meta_features = pd.DataFrame([meta_features_dict])
categorical_cols = ["mode", "icc_present", "has_iCCP", "has_tEXt"]

for col in categorical_cols:
    meta_features[col] = meta_features[col].astype(str)
    
meta_prob = meta_model.predict_proba(meta_features)[0,1]

w_pixel = 0.7
w_meta = 0.3

fusion_prob = (w_pixel * pixel_prob) + (w_meta * meta_prob)
print("Classifier: "+str(pixel_prob))
print("Forensic: "+str(meta_prob))
print("Fusion: " +str(fusion_prob))
label = "real" if fusion_prob >= 0.5 else "ai"

print(label)
