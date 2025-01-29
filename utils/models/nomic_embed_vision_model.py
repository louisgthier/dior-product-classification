# fashion_clip_model.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from .base_model import BaseModel
from PIL import Image

class NomicEmbedVisionModel(BaseModel):
    def load_model(self):
        """
        Load the Fashion-CLIP model.
        """
        model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        return model.to(self.device)
    
    def preprocess_for_model(self, pil_image):
        """
        Preprocess input image (and optionally texts) for the Fashion-CLIP model.

        Args:
            pil_image (PIL.Image.Image): The input image to preprocess.
            texts (list of str, optional): List of text descriptions to process alongside the image.

        Returns:
            dict: A dictionary of preprocessed inputs ready for the model.
        """
        
        inputs = self.processor(pil_image, return_tensors="pt")
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    def extract_features(self, pil_image):
        """
        Extract image features using the Fashion-CLIP model.

        Args:
            pil_image (PIL.Image.Image): The input image.

        Returns:
            numpy.ndarray: The extracted image features as a flattened NumPy array.
        """
        self.model.eval()
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            # Extract image features
            img_emb = self.model(**inputs).last_hidden_state
            # print(f"img_emb shape: {img_emb.shape}")
            # img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
            # print(f"img_embeddings shape: {img_embeddings.shape}")
        # return img_embeddings.cpu().numpy()
        
        return img_emb.cpu().numpy()