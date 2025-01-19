import os
import base64
from glob import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import modal
from utils.preprocessing import preprocess_image
import base64
import random

random.seed(42)

ALL_IMAGES = True

# Environment configuration for TRELLIS output directory
cache_dir = os.path.join('.cache', 'TRELLIS')
os.makedirs(cache_dir, exist_ok=True)

# Configure paths and directories
dam_dir = 'data/DAM'
labels_csv = 'labels/handmade_test_labels.csv'

# Load labels CSV and extract references
labels_df = pd.read_csv(labels_csv)

# Create a set of all references from the CSV (split on '/')
references = set()
for refs in labels_df['reference'].dropna():
    for ref in refs.split('/'):
        ref = ref.strip()
        if ref:
            references.add(ref)

print(f"Total unique references from CSV: {len(references)}")

# Gather all DAM image file paths
extensions = ['*.jpg', '*.jpeg', '*.png']
dam_images = []
for ext in extensions:
    dam_images.extend(glob(os.path.join(dam_dir, ext)))
dam_images.sort()

# Filter DAM images based on whether their filename (without extension) is in references
filtered_dam_images = []
for img_path in dam_images:
    file_name = os.path.basename(img_path)
    name_without_ext, _ = os.path.splitext(file_name)
    if name_without_ext in references or ALL_IMAGES:
        filtered_dam_images.append(img_path)

print(f"Total images to process after filtering: {len(filtered_dam_images)}")

def generate_and_save_model(dam_id: str, file_path: str):
    # Lookup the model from Modal registry
    model = modal.Cls.lookup("trellis-3d-generation", "Model")()  # Adjust lookup if needed

    # Preprocess the image
    image = preprocess_image(file_path)

    # Call the model without generating videos
    result = model.generate_3d.remote(
        image,
        seed=1,
        render_gaussian_video=False,
        render_rf_video=False,
        render_mesh_video=False,
    )  # Wait for result synchronously; adjust if needed for async

    # If GLB data exists, write it to file
    if "glb" in result:
        glb_path = os.path.join(".cache", "TRELLIS", f"{dam_id}.glb")
        with open(glb_path, "wb") as f:
            f.write(base64.b64decode(result["glb"]))
        print(f"Saved {glb_path}")
    else:
        print(f"No GLB output for {dam_id}")
        
# Shuffle the filtered DAM images
random.shuffle(filtered_dam_images)
print(f"Shuffled images for processing")

def main():
    with modal.enable_output():
        # Loop over each filtered DAM image and process with Modal
        for idx, img_path in tqdm(enumerate(filtered_dam_images), desc="Generating 3D models via Modal"):
            # if idx >= 2:
            #     break
            
            file_name = os.path.basename(img_path)
            dam_id, _ = os.path.splitext(file_name)
            output_glb_path = os.path.join(cache_dir, f"{dam_id}.glb")

            # Skip processing if GLB file already exists
            if os.path.exists(output_glb_path):
                continue

            # Invoke the Modal function for each image
            generate_and_save_model(dam_id, img_path)
            
if __name__ == "__main__":
    main()