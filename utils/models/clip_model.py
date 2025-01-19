import torch
from transformers import CLIPProcessor, CLIPModel

# Preprocessing for CLIP
def preprocess_for_model(pil_image):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(images=pil_image, return_tensors="pt")
    return inputs

# Feature extraction for CLIP
def extract_features(pil_image, model):
    inputs = preprocess_for_model(pil_image)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        # Extract image features
        features = model.get_image_features(**inputs)
    return features.cpu().numpy().flatten()

# Load the CLIP model
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return model
