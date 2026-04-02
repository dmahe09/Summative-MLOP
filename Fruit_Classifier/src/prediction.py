"""
Prediction module for fruit classification
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(model_path, img_path, img_size=(100, 100)):
    """
    Predict the class of a single image
    
    Args:
        model_path: Path to the trained model file
        img_path: Path to the image to predict
        img_size: Tuple of (height, width) for resizing
        
    Returns:
        class_idx: Index of predicted class
        prediction: Full prediction array (probabilities for all classes)
        confidence: Confidence score for predicted class
    """
    # Load model
    model = load_model(model_path)
    
    # Load and preprocess image
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    
    return class_idx, prediction[0], confidence


def predict_from_array(model_path, img_array):
    """
    Predict from a preprocessed image array
    
    Args:
        model_path: Path to the trained model
        img_array: Preprocessed image array
        
    Returns:
        class_idx: Index of predicted class
        prediction: Full prediction array
        confidence: Confidence score
    """
    model = load_model(model_path)
    
    # Ensure correct shape
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    
    return class_idx, prediction[0], confidence


def batch_predict(model_path, image_paths, img_size=(100, 100)):
    """
    Predict multiple images at once
    
    Args:
        model_path: Path to the trained model
        image_paths: List of image file paths
        img_size: Tuple of (height, width)
        
    Returns:
        List of tuples (class_idx, confidence) for each image
    """
    model = load_model(model_path)
    results = []
    
    for img_path in image_paths:
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        results.append((class_idx, confidence))
    
    return results