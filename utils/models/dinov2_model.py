import torch
from torchvision import transforms
from torch.hub import load as torch_hub_load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_for_model(pil_image):
    """
    Preprocess the input image for DINOv2.
    """
    preprocess_dino = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    return preprocess_dino(pil_image).unsqueeze(0).to(device)

def extract_features(pil_image, model):
    """
    Extract features for an image using DINOv2.
    """
    model.eval()

    img_tensor = preprocess_for_model(pil_image)

    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().flatten()

def load_model():
    """
    Load the DINOv2 model from PyTorch Hub.
    """
    # Load model and move it to the correct device
    model = torch_hub_load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    return model
