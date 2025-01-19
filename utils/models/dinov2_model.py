# dinov2_model.py
import torch
from transformers import AutoImageProcessor, AutoModel
from .base_model import BaseModel

class DinoV2Model(BaseModel):
    def load_model(self):
        """
        Load the DINOv2 model and its corresponding image processor from Hugging Face.
        """
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        return model.to(self.device)

    def preprocess_for_model(self, pil_image):
        """
        Preprocess the input image using the Hugging Face image processor.
        """
        inputs = self.processor(images=pil_image, return_tensors="pt", use_fast=True)
        # Move inputs to the correct device
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_features(self, pil_image):
        """
        Extract features for an image using the loaded DINOv2 model.
        """
        self.model.eval()
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the last_hidden_state as the feature representation
        last_hidden_states = outputs.last_hidden_state
        # return last_hidden_states.mean(dim=1).cpu().numpy().flatten()  # Global average pooling
        return last_hidden_states.cpu().numpy()