# vit_model.py
import torch
from transformers import ViTImageProcessor, ViTModel
from .base_model import BaseModel

class GoogleViTModel(BaseModel):
    def load_model(self):
        model_size = "large"
        in21k = False
        name = f"google/vit-{model_size}-patch16-224{'-in21k' if in21k else ''}"
        self.processor = ViTImageProcessor.from_pretrained(name)
        self.model_name = type(self).__name__ + "_" + model_size + ("_in21k" if in21k else "")
        return ViTModel.from_pretrained(name).to(self.device)

    def preprocess_for_model(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt", use_fast=True)
        # Move inputs to the correct device
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_features(self, pil_image):
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)
        # last_hidden_states = last_hidden_states[:, 1:, :]  # Exclude CLS token
        # pooled_embedding = last_hidden_states.mean(dim=1)  # Global average pooling
        # pooled_embedding = last_hidden_states[:, -1, :] # Take only the last token's hidden state
        # return pooled_embedding.cpu().numpy().flatten()
        return last_hidden_states.cpu().numpy()