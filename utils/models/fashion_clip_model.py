# fashion_clip_model.py
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from .base_model import BaseModel
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
class FashionCLIPModel(BaseModel):
    def load_model(self):
        """
        Load the Fashion-CLIP model.
        """
        self.fclip = FashionCLIP('fashion-clip')
        return self.fclip
        
        # Load the Fashion-CLIP model
        # model = FashionCLIP('fashion-clip')
        # return model
    
    def preprocess_for_model(self, pil_image, texts=None):
        """
        Preprocess input image (and optionally texts) for the Fashion-CLIP model.

        Args:
            pil_image (PIL.Image.Image): The input image to preprocess.
            texts (list of str, optional): List of text descriptions to process alongside the image.

        Returns:
            dict: A dictionary of preprocessed inputs ready for the model.
        """
        
        # Resize the image to (224, 224) as per Fashion-CLIP requirements
        # resized_image = pil_image.resize((224, 224))
        
        if texts is not None:
            inputs = self.processor(text=texts, images=pil_image, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt", padding=True)
        
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
        
        single_image = type(pil_image) != list
        
        if single_image:
            pil_image = [pil_image]
        
        image_embeddings = self.fclip.encode_images(pil_image, batch_size=32)
        
        if single_image:
            image_embeddings = image_embeddings[0]
        
        # image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        return image_embeddings
        
        # if type(pil_image) != list:
        #     pil_image = [pil_image]
        # image_embeddings = self.model.encode_images(pil_image, batch_size=32)
        # if type(pil_image) is not list:
        #     return image_embeddings[0]
        # return image_embeddings
    