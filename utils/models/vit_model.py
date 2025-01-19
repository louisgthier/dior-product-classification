import torch
from transformers import ViTImageProcessor, ViTModel

def preprocess_for_model(pil_image):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    inputs = processor(images=pil_image, return_tensors="pt")
    return inputs

def extract_features(pil_image, model):
    inputs = preprocess_for_model(pil_image)
    inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state  # shape: (1, sequence_length, hidden_size)
    token_embeddings = last_hidden_states[:, 1:, :]  # Exclude the CLS token
    pooled_embedding = token_embeddings.mean(dim=1)  # Global average pooling
    return pooled_embedding.cpu().numpy().flatten()

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    return model
