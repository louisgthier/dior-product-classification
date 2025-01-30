# app.py

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from tqdm import tqdm
import pandas as pd
from glob import glob
import PIL
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import sys
import gradio as gr
import tempfile

from utils.preprocessing import preprocess_image
import utils.preprocessing
import importlib
importlib.reload(utils.preprocessing)

import matplotlib.pyplot as plt
from PIL import Image, ImageOps as PIL_ImageOps

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
import importlib
import json
import faiss  # Import FAISS
from sklearn.decomposition import PCA  # Import PCA

import utils.models
import utils.models.nomic_embed_vision_model
importlib.reload(utils.models.base_model)
importlib.reload(utils.models.dinov2_model)
importlib.reload(utils.models.facebook_vitmsn_model)
importlib.reload(utils.models.google_vit_model)
importlib.reload(utils.models.microsoft_resnet_model)
importlib.reload(utils.models.openai_clip_model)
importlib.reload(utils.models.fashion_clip_model)
importlib.reload(utils.models.nomic_embed_vision_model)
importlib.reload(utils.models)
import utils.models
from utils.models import DinoV2Model, FacebookViTMSNModel, GoogleViTModel, MicrosoftResNetModel, OpenAIClipModel, FashionCLIPModel, NomicEmbedVisionModel

# ---------------------------- Configuration ----------------------------

# Define paths to necessary files and directories
BASE_CACHE_DIR = '.cache'
BASE_PRECOMPUTED_DIR = "precomputed"
DAM_FEATURE_PATHS_PICKLE = os.path.join(BASE_PRECOMPUTED_DIR, 'paths', 'dam_feature_paths.pkl')  # Update as needed
FEATURE_SELECTION_COEFFICIENTS_PICKLE = os.path.join(BASE_PRECOMPUTED_DIR, 'coefficients', 'feature_selection_coefficients.pkl')  # Update as needed
MERGING_COEFFICIENTS_PICKLE = os.path.join(BASE_PRECOMPUTED_DIR, 'coefficients', 'merging_coefficients.pkl')  # Update as needed
LABELS_CSV_PATH = 'labels/handmade_test_labels.csv'  # Update as needed
EMBEDDING_AGGREGRATION_METHOD = 'max'  # Update as needed

# ---------------------------- Device Setup ----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 
                      'cpu')
print(f"Using device: {device}")

# ---------------------------- Model Setup ----------------------------

# Initialize and load the model
model = GoogleViTModel()
    
FAISS_INDEX_PATH = os.path.join(BASE_PRECOMPUTED_DIR, 'faiss', f'dam_features-{model.model_name}-RMBG_2-3d.index')  # Update as needed

# ---------------------------- Feature Aggregation ----------------------------

def aggregate_embedding(embedding):
    if len(embedding.shape) >= 2:
        # Flatten to (num_patches, feature_dim)
        embedding = embedding.squeeze()
        # Apply PCA transformation
        
        # Aggregate by averaging
        if EMBEDDING_AGGREGRATION_METHOD == 'mean':
            embedding = np.mean(embedding, axis=0).reshape(1, -1)  # Shape: (1, n_components)
        elif EMBEDDING_AGGREGRATION_METHOD == 'CLS':
            embedding = embedding[0, :].reshape(1, -1)  # Shape: (1, n_components) # Take only the CLS token
        elif EMBEDDING_AGGREGRATION_METHOD == 'sum':
            embedding = np.sum(embedding, axis=0).reshape(1, -1)  # Shape: (1, n_components)
        elif EMBEDDING_AGGREGRATION_METHOD == 'max':
            embedding = np.max(embedding, axis=0).reshape(1, -1)  # Shape: (1, n_components)
            
    return embedding

# ---------------------------- Load FAISS Index and Mappings ----------------------------

# Initialize placeholders
index = None
dam_feature_paths = []
feature_selection_coefficients = None
merging_coefficients = None

# Load FAISS index
try:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
    
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"FAISS index loaded from {FAISS_INDEX_PATH}. Number of vectors: {index.ntotal}, Dimension: {index.d}")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    sys.exit(1)

# Load DAM feature paths
try:
    if not os.path.exists(DAM_FEATURE_PATHS_PICKLE):
        raise FileNotFoundError(f"DAM feature paths pickle not found at {DAM_FEATURE_PATHS_PICKLE}")
    
    with open(DAM_FEATURE_PATHS_PICKLE, 'rb') as f:
        dam_feature_paths = pickle.load(f)
    
    print(f"Loaded {len(dam_feature_paths)} DAM feature paths.")
except Exception as e:
    print(f"Error loading DAM feature paths: {e}")
    sys.exit(1)

# Load feature selection coefficients
try:
    if False and os.path.exists(FEATURE_SELECTION_COEFFICIENTS_PICKLE):
        with open(FEATURE_SELECTION_COEFFICIENTS_PICKLE, 'rb') as f:
            feature_selection_coefficients = pickle.load(f)
        print("Loaded feature selection coefficients.")
    else:
        feature_selection_coefficients = np.ones(index.d, dtype=np.float32)
        print("Feature selection coefficients not found. Using default ones.")
except Exception as e:
    print(f"Error loading feature selection coefficients: {e}")
    sys.exit(1)

# Load merging coefficients (if applicable)
try:
    if False and os.path.exists(MERGING_COEFFICIENTS_PICKLE):
        with open(MERGING_COEFFICIENTS_PICKLE, 'rb') as f:
            merging_coefficients = pickle.load(f)
        print("Loaded merging coefficients.")
    else:
        merging_coefficients = np.ones(index.d, dtype=np.float32) * 0.5
        print("Merging coefficients not found. Using default ones.")
except Exception as e:
    print(f"Error loading merging coefficients: {e}")
    sys.exit(1)

# ---------------------------- Load Labels ----------------------------

try:
    if not os.path.exists(LABELS_CSV_PATH):
        raise FileNotFoundError(f"Labels CSV not found at {LABELS_CSV_PATH}")
    
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    
    # Create a dictionary mapping each test image filename to a list of reference labels
    labels_dict = {}
    for _, row in labels_df.iterrows():
        image_name = row['image'].strip()
        references = [ref.strip() for ref in str(row['reference']).split('/') if ref.strip() and ref.strip() != '?']
        labels_dict[image_name] = references
    
    print(f"Loaded labels for {len(labels_dict)} images.")
except Exception as e:
    print(f"Error loading labels: {e}")
    sys.exit(1)

# ---------------------------- Gradio Interface Function ----------------------------

def find_similar_images(uploaded_image, top_n=5):
    """
    Process the uploaded image, extract features, search FAISS index, and return top matches.
    Includes logging to identify where a crash might occur.
    """
    try:
        print("Starting similarity search.")
        if uploaded_image is None:
            print("No image uploaded.")
            return "No image uploaded.", None

        # Save to a temporary file because preprocess_image expects a file path
        with tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False) as temp_image:
            uploaded_image.save(temp_image.name)
            temp_image_path = temp_image.name

        # Open and preprocess the image
        print("Opening uploaded image.")
        processed_image = preprocess_image(temp_image_path)
        if processed_image is None:
            print("Image preprocessing failed.")
            return "Image preprocessing failed.", None

        # Extract features using the model
        print("Extracting features using the model.")
        with torch.no_grad():
            features = model.extract_features(processed_image)  # Assuming the model has an extract_features method
            if features is None:
                print("Feature extraction returned None.")
                return "Feature extraction failed.", None
            aggregated_features = aggregate_embedding(features)
            if aggregated_features is None:
                print("Feature aggregation returned None.")
                return "Feature aggregation failed.", None

        # Apply feature selection coefficients
        print("Applying feature selection coefficients.")
        aggregated_features *= feature_selection_coefficients

        # Convert to float32
        print("Converting query vector to float32.")
        query_vector = aggregated_features.astype(np.float32)

        # Ensure the array is C-contiguous
        if not query_vector.flags['C_CONTIGUOUS']:
            print("Converting query vector to C-contiguous array.")
            query_vector = np.ascontiguousarray(query_vector)

        # Normalize if using cosine similarity
        if isinstance(index, faiss.IndexFlatIP):
            print("Normalizing query vector for cosine similarity.")
            norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_vector = query_vector / norms

        # Check if top_n exceeds index size
        if top_n > index.ntotal:
            print(f"Requested top_n ({top_n}) exceeds index size ({index.ntotal}). Adjusting top_n to {index.ntotal}.")
            top_n = index.ntotal

        # Perform FAISS search
        print(f"Performing FAISS search for top {top_n} matches.")
        distances, indices = index.search(query_vector, top_n * 3)
        print("FAISS search completed.")

        # Retrieve matching DAM IDs
        print("Retrieving matching DAM IDs.")
        matches = []
        found_ids = set()
        for idx in indices[0]:
            if idx < 0 or idx >= len(dam_feature_paths):
                print(f"Invalid index retrieved from FAISS: {idx}")
                matches.append("Invalid Index")
                continue
            dam_path = dam_feature_paths[idx]
            dam_id = os.path.basename(dam_path).split('.')[0].split("-")[0]  # Extract DAM ID from path
            if dam_id in found_ids:
                continue
            found_ids.add(dam_id)
            matches.append(dam_id)
        
        matches = matches[:top_n]  # Limit to top_n matches

        # Prepare display images and DAM IDs
        print("Preparing matched images and DAM IDs for display.")
        match_images = []
        match_ids = []
        for dam_id in matches:
            # Construct the path to the DAM image
            dam_image_path = os.path.join('data', 'DAM', f"{dam_id}.jpeg")  # Update as needed
            if os.path.exists(dam_image_path):
                try:
                    dam_image = Image.open(dam_image_path).convert('RGB')
                except Exception as e:
                    print(f"Error opening DAM image {dam_image_path}: {e}")
                    dam_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
            else:
                print(f"DAM image not found at {dam_image_path}. Using placeholder.")
                # Placeholder image if not found
                dam_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
            match_images.append(dam_image)
            match_ids.append(dam_id)

        print("Similarity search completed successfully.")
        
        # Path to the .glb file of the best match
        best_match_glb_path = os.path.join(BASE_CACHE_DIR, 'TRELLIS', f"{matches[0]}.glb")
        if not os.path.exists(best_match_glb_path):
            best_match_glb_path = None

        return match_images, "\n".join(match_ids), best_match_glb_path

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}", None

# ---------------------------- Gradio App Setup ----------------------------

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Similarity Search using FAISS and GoogleViTModel")
    gr.Markdown("Upload an image to find the most similar DAM images.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            top_n_slider = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Number of Top Matches")
            search_button = gr.Button("Search")
            progress_text = gr.Textbox(label="Progress", interactive=False)
        with gr.Column():
            output_gallery = gr.Gallery(label="Top Matches")
            dam_ids_output = gr.Textbox(label="DAM IDs", interactive=False)
            best_match_3d = gr.Model3D(label="Best Match 3D Model")

    def on_search(image, top_n):
        progress = ""
        try:
            if image is None:
                progress = "Please upload an image."
                return progress, None, None, None
            progress += "Starting similarity search...\n"
            match_images, match_ids, best_match_glb_path = find_similar_images(image, top_n=top_n)
            if isinstance(match_images, str):
                progress += match_images
                return progress, None, None, None
            display_images = []
            display_ids = []
            for img, dam_id in zip(match_images, match_ids.split('\n')):
                display_images.append(img)
                display_ids.append(dam_id)
            progress += "Similarity search completed successfully."
            return progress, display_images, "\n".join(display_ids), best_match_glb_path
        except Exception as e:
            progress += f"An error occurred during the search: {e}"
            return progress, None, None, None

    search_button.click(fn=on_search, inputs=[image_input, top_n_slider], outputs=[progress_text, output_gallery, dam_ids_output, best_match_3d])
# ---------------------------- Launch Gradio App ----------------------------

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch()