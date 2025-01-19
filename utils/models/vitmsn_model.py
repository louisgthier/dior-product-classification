import torch
from transformers import ViTImageProcessor, ViTMSNModel

def preprocess_for_model(pil_image):
    processor = ViTImageProcessor.from_pretrained("facebook/vit-msn-base")
    inputs = processor(images=pil_image, return_tensors="pt")
    return inputs

def extract_features(pil_image, model):
    inputs = preprocess_for_model(pil_image)
    inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
    token_embeddings = last_hidden_states[:, 1:, :]  # Exclude CLS token
    pooled_embedding = token_embeddings.mean(dim=1)  # Global average pooling
    return pooled_embedding.cpu().numpy().flatten()

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMSNModel.from_pretrained("facebook/vit-msn-base").to(device)
    return model
