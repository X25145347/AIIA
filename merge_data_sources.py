import os
import csv
import json
import shutil
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

REAL_IMAGES_DIR = "./ai-dataset/real/"
AI_IMAGES_DIR = "./ai-dataset/ai"
OUTPUT_DIR = "./merged_data_sources/"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images/")
exif_json = os.path.join(OUTPUT_DIR, "exif_metadata.json")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.csv")
TARGET_SIZE = (512, 512) 

# ---------------------------------------------------------
# Ensure output folders exist
# ---------------------------------------------------------

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helper function to extract exif data from image before 
# reformatting the image
# ---------------------------------------------------------
def extract_exif(image_path):
	try:
		img = Image.open(image_path)
		exif_data = img.getexif()
		
		if not exif_data:
			return None
		readable = {}
		for tag_id, value in exif_data.items():
			tag = TAGS.get(tag_id, tag_id)
			readable[tag] = value
		return readable
	except Exception:
		return None


# ---------------------------------------------------------
# Helper function to copy images, resize the images and 
# create labels
# ---------------------------------------------------------

def process_folder(source_dir, label):
    entries = []
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Warning: {source_dir} does not exist.")
        return entries

    for img in source_path.iterdir():
        if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            dest = Path(OUTPUT_IMAGES_DIR) / img.name
            exif = extract_exif(img) if label == "real" else None
            resize_image(img, dest, TARGET_SIZE)
            exif_stat = ""
            metadata_loss = False
            if label == "real" and exif == None:
                exif_stat = "missing"
                metadata_loss = True
            elif label == "ai":
                exif_stat = "None"
                metadata_loss = True
            else:
                exif_stat = "Present"
                metadata_loss = False
            entries.append({
                "filename": img.name,
                "source": label,
                "exif_present": exif is not None,
                "exif_status": exif_stat,
                "metadata_loss": metadata_loss,
                "exif": exif
            })

    return entries

# ---------------------------------------------------------
# Helper function to resize image
# ---------------------------------------------------------

def resize_image(input_path, output_path, size):
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        img.save(output_path)
        print(f"Resized: {input_path.name}")
    except Exception as e:
        print(f"Error resizing {input_path.name}: {e}")

# ---------------------------------------------------------
# Process real + AI images
# ---------------------------------------------------------

real_entries = process_folder(REAL_IMAGES_DIR, "real")
ai_entries = process_folder(AI_IMAGES_DIR, "ai")

# ---------------------------------------------------------
# Merge real + AI images
# ---------------------------------------------------------

all_entries = real_entries + ai_entries

# ---------------------------------------------------------
# Write exif_metadata.json
# ---------------------------------------------------------

with open(exif_json, "w") as f:
    json.dump({e["filename"]: e["exif"] for e in all_entries}, f, indent=4)
    
# ---------------------------------------------------------
# Write labels.csv
# ---------------------------------------------------------
    
with open(LABELS_FILE, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_entries[0].keys())
    writer.writeheader()
    writer.writerows(all_entries)

print(f"Merged {len(all_entries)} images into {OUTPUT_IMAGES_DIR}")
print(f"Labels saved to {LABELS_FILE}")
