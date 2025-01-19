import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

def preprocess_for_model(pil_image):
    target_size = (224, 224)
    image_array = keras_image.img_to_array(pil_image)
    img_resized = keras_image.smart_resize(image_array, target_size)
    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

def extract_features(pil_image, model):
    preprocessed = preprocess_for_model(pil_image)
    features = model.predict(preprocessed, verbose=0)
    return features.flatten()

def load_model():
    return tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")
