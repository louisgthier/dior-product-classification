# fashion_clip_model.py
import torch
from transformers import CLIPProcessor, CLIPModel
from .base_model import BaseModel
from PIL import Image
# from fashion_clip.fashion_clip import FashionCLIP

class FashionCLIPModel(BaseModel):
    def load_model(self):
        """
        Load the Fashion-CLIP model.
        """
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        return model.to(self.device)
        
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
        self.model.eval()
        inputs = self.preprocess_for_model(pil_image)
        with torch.no_grad():
            # Extract image features
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()
        
        # if type(pil_image) != list:
        #     pil_image = [pil_image]
        # image_embeddings = self.model.encode_images(pil_image, batch_size=32)
        # if type(pil_image) is not list:
        #     return image_embeddings[0]
        # return image_embeddings
    
    def extract_text_features(self, texts):
        """
        Extract text features using the Fashion-CLIP model.

        Args:
            texts (list of str): List of text descriptions.

        Returns:
            numpy.ndarray: The extracted text features as a flattened NumPy array.
        """
        self.model.eval()
        # Dummy image to satisfy the processor requirements
        dummy_image = Image.new("RGB", (224, 224))
        inputs = self.preprocess_for_model(dummy_image, texts=texts)
        with torch.no_grad():
            # Extract text features
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy().flatten()
    
    def compute_similarity(self, pil_image, texts):
        """
        Compute similarity scores between an image and a list of texts.

        Args:
            pil_image (PIL.Image.Image): The input image.
            texts (list of str): List of text descriptions.

        Returns:
            numpy.ndarray: Softmax probabilities representing similarity scores.
        """
        self.model.eval()
        inputs = self.preprocess_for_model(pil_image, texts=texts)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Image-text similarity scores
            probs = logits_per_image.softmax(dim=1)
        return probs.cpu().numpy()