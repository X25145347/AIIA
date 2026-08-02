import os
from PIL import Image
import jpegio as jio  
import png

def extract_metadata_features(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    img = Image.open(image_path)
    metadata = get_generic_metadata(img, image_path)
    # JPEG metadata
    if ext in [".jpg", ".jpeg"]:
        metadata.update(get_jpeg_metadata(image_path))

    # PNG metadata
    if ext == ".png":
        metadata.update(get_png_metadata(image_path))

    return metadata
    
# Gets the Generic/Common image details from the image like height and width
def get_generic_metadata(img, path):
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

# Gets the JPEG/JPG specific metadata
def get_jpeg_metadata(path):
    jpeg = jio.read(path)
    qtables = jpeg.quant_tables  # list of arrays
    metadata = {
        "num_qtables": len(qtables),
        "qt_mean_0": qtables[0].mean() if len(qtables) > 0 else 0,
        "qt_std_0": qtables[0].std() if len(qtables) > 0 else 0,
        "has_iCCP": 0,
        "has_tEXt": 0,
        "num_chunks": 0
    }
    return metadata

# Gets the PNG specific metadata
def get_png_metadata(path):
    reader = png.Reader(filename=path)
    chunks = list(reader.chunks())
    chunk_types = [ct.decode("ascii") if isinstance(ct, bytes) else ct for ct, _ in chunks]
    metadata = {
        "num_chunks": len(chunks),
        "has_iCCP": int("iCCP" in chunk_types),
        "has_tEXt": int("tEXt" in chunk_types),
        "qt_std_0": 0,
        "qt_mean_0": 0,
        "num_qtables": 0
    }

    return metadata

def get_images_metadata(folder):
    full_path = "./ai-dataset/"+folder+"/"
    files = os.listdir(full_path)
    metadata_rows = []
    for image_name in files:
        image_path = full_path+image_name
        metadata = {"filename":image_name, "label":folder}
        metadata.update(extract_metadata_features(image_path))
        metadata_rows.append(metadata)
    return metadata_rows
