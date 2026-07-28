from PIL import Image
import numpy as np
import piexif
import jpegio as jio  # JPEG quantization tables
import os
import png
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from joblib import dump

def get_generic_metadata(path):
    img = Image.open(path)
    width, height = img.size
    file_size = os.path.getsize(path)

    bytes_per_pixel = file_size / (width * height)

    metadata = {
        "width": width,
        "height": height,
        "bytes_per_pixel": bytes_per_pixel,
        "mode": img.mode,
        "icc_present": "icc_profile" in img.info,
    }
    return metadata

def get_jpeg_metadata(path):
    jpeg = jio.read(path)
    qtables = jpeg.quant_tables  # list of arrays
    metadata = {
        "num_qtables": len(qtables),
        "qt_mean_0": qtables[0].mean() if len(qtables) > 0 else 0,
        "qt_std_0": qtables[0].std() if len(qtables) > 0 else 0,
    }
    metadata["has_iCCP"]= 0
    metadata["has_tEXt"]= 0
    metadata["num_chunks"]= 0
    return metadata

def get_png_metadata(path):
    reader = png.Reader(filename=path)
    chunks = list(reader.chunks())

    chunk_types = [ct.decode("ascii") if isinstance(ct, bytes) else ct
                   for ct, _ in chunks]

    metadata = {
        "num_chunks": len(chunks),
        "has_iCCP": "iCCP" in chunk_types,
        "has_tEXt": "tEXt" in chunk_types,
    }
    return metadata

def get_images_metadata(folder):
    full_path = "./ai-dataset/"+folder+"/"
    files = os.listdir(full_path)
    print(len(files))
    metadata_rows = []
    for image_name in files:
        image_path = full_path+image_name
        metadata = get_generic_metadata(image_path)
        metadata["filename"] = image_name
        metadata["label"] = folder
        if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
           metadata.update(get_jpeg_metadata(image_path))
        elif image_path.endswith(".png"):
           metadata.update(get_png_metadata(image_path))
        metadata_rows.append(metadata)
    return metadata_rows

metadata_master_rows = []
metadata_master_rows = get_images_metadata("real")
metadata_master_rows.extend(get_images_metadata("ai"))

df = pd.DataFrame(metadata_master_rows)

categorical_cols = ["mode", "icc_present", "has_iCCP", "has_tEXt"]
numeric_cols = ["width", "height", "bytes_per_pixel", "num_qtables", "num_chunks", "qt_mean_0", "qt_std_0"]

# Build preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(
            handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# Build full model pipeline
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

X = df[categorical_cols + numeric_cols]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1:", f1_score(y_val, y_pred, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
precision = precision_score(y_val, y_pred, average="macro")
recall = recall_score(y_val, y_pred, average="macro")

print("Precision:", precision)
print("Recall:", recall)

dump(clf, "metadata_forensic_model.joblib")