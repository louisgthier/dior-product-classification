# resnet_model.py
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from .base_model import BaseModel

class MicrosoftResNetModel(BaseModel):
    def load_model(self):
        """
        Load the ResNet model and its corresponding image processor from Hugging Face.
        """
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        return model.to(self.device)

    def preprocess_for_model(self, pil_image):
        """
        Preprocess the input image using the Hugging Face image processor.
        """
        inputs = self.processor(pil_image, return_tensors="pt", use_fast=True)
        # Move inputs to the correct device
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_features(self, pil_image):
        """
        Extract features (logits) for an image using the loaded ResNet model.
        """
        self.model.eval()
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        return logits.cpu().numpy().flatten()