import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
csv_path = "./merged_data_sources/labels.csv"          # your CSV file
source_dir = "./merged_data_sources/images"    # merged folder with all images

train_real_dir = "./training_data/real/"
val_real_dir = "./validate_data/real/"
test_real_dir = "./test_data/real/"

train_ai_dir = "./training_data/ai/"
val_ai_dir = "./validate_data/ai/"
test_ai_dir = "./test_data/ai/"

# -----------------------------
# Create output folders
# -----------------------------
os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_ai_dir, exist_ok=True)
os.makedirs(val_real_dir, exist_ok=True)
os.makedirs(val_ai_dir, exist_ok=True)
os.makedirs(test_real_dir, exist_ok=True)
os.makedirs(test_ai_dir, exist_ok=True)

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_path)

# Expect columns: filename, is_ai
ai_files = df[df["source"] == "ai"]["filename"].tolist()

print(ai_files)

real_files = df[df["source"] == "real"]["filename"].tolist()

print(real_files)

# -----------------------------
# Split AI images: 70/15/15
# -----------------------------
ai_train, ai_temp = train_test_split(ai_files, test_size=0.30, random_state=42)
ai_val, ai_test = train_test_split(ai_temp, test_size=0.50, random_state=42)

# -----------------------------
# Split Real images: 70/15/15
# -----------------------------
real_train, real_temp = train_test_split(real_files, test_size=0.30, random_state=42)
real_val, real_test = train_test_split(real_temp, test_size=0.50, random_state=42)

# -----------------------------
# Copy files
# -----------------------------
def copy_files(file_list, cls, split):
    for f in file_list:
        src = os.path.join(source_dir, f)
        dst = os.path.join("./", split, cls, f)

        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"WARNING: File not found: {src}")

copy_files(ai_train, "ai", "training_data")
copy_files(ai_val, "ai", "validate_data")
copy_files(ai_test, "ai", "test_data")

copy_files(real_train, "real", "training_data")
copy_files(real_val, "real", "validate_data")
copy_files(real_test, "real", "test_data")

print("70/15/15 dataset split complete.")
