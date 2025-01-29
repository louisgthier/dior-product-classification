# base_model.py
from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.model_name = type(self).__name__
        self.model = self.load_model()
        print(f"Model loaded on device: {self.device}")

    @staticmethod
    def check_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @abstractmethod
    def preprocess_for_model(self, pil_image):
        """
        Preprocess the input image for the specific model.
        """
        pass

    @abstractmethod
    def extract_features(self, pil_image):
        """
        Extract features for an image using the specific model.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Load and return the model.
        """
        pass