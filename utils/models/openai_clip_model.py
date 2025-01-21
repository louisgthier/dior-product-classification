# clip_model.py
import torch
from transformers import CLIPProcessor, CLIPModel
from .base_model import BaseModel

class OpenAIClipModel(BaseModel):
    def load_model(self):
        """
        Load the CLIP model.
        """
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

    def preprocess_for_model(self, pil_image):
        """
        Preprocess input image for CLIP.
        """
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=pil_image, return_tensors="pt", use_fast=True)
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def extract_features(self, pil_image):
        """
        Extract image features using the CLIP model.
        """
        self.model.eval()
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy().flatten()