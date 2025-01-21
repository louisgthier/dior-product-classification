# vitmsn_model.py
import torch
from transformers import ViTImageProcessor, ViTMSNModel
from .base_model import BaseModel

class FacebookViTMSNModel(BaseModel):
    def load_model(self):
        return ViTMSNModel.from_pretrained("facebook/vit-msn-base").to(self.device)

    def preprocess_for_model(self, pil_image):
        processor = ViTImageProcessor.from_pretrained("facebook/vit-msn-base")
        inputs = processor(images=pil_image, return_tensors="pt")
        # Move inputs to the correct device
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_features(self, pil_image):
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)
        # token_embeddings = last_hidden_states[:, 1:, :]  # Exclude CLS token
        # pooled_embedding = token_embeddings.mean(dim=1)  # Global average pooling
        # return pooled_embedding.cpu().numpy().flatten()
        return last_hidden_states.cpu().numpy()