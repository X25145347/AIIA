import torch
from joblib import load
import torch.nn as nn
from torchvision import models, transforms 
import pandas as pd
import gradio as gr
from PIL import Image
import AIIA_Common as aiia_com

def run_fusion(image_path):
    
    # Pixel model
    pil_img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image_tensor = preprocess(pil_img).unsqueeze(0)
    state_dict = torch.load("efficientnet_ai_vs_real.pth", map_location="cpu")
    pixel_model =  EfficientNetModel()
    pixel_model.load_state_dict(state_dict)
    pixel_model.eval()

    with torch.no_grad():
        outputs = pixel_model(image_tensor)
        pixel_prob = torch.softmax(outputs, dim=1)[0,1].item()

    # Metadata model
    meta_features_dict = aiia_com.extract_metadata_features(image_path)
    meta_features = pd.DataFrame([meta_features_dict])
    # Load metadata model (RandomForest)
    meta_model = load("metadata_forensic_model.joblib")
    categorical_cols = ["mode", "icc_present", "has_iCCP", "has_tEXt"]

    for col in categorical_cols:
        meta_features[col] = meta_features[col].astype(str)
    
    meta_prob = meta_model.predict_proba(meta_features)[0,1]
    fusion_prob = 0.7 * pixel_prob + 0.3 * meta_prob
    label = "Real" if fusion_prob >= 0.5 else "AI-generated"

    return label
    
class EfficientNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
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


ui = gr.Interface(
    fn=run_fusion,
    inputs=gr.Image(type="filepath"),
    outputs="text"
)

ui.launch()
