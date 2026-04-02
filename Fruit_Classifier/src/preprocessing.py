"""
Data preprocessing module for fruit classification
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def create_data_generators(train_dir, test_dir, img_size=(100, 100), batch_size=32):
    """
    Create training and testing data generators with augmentation
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        img_size: Tuple of (height, width) for resizing images
        batch_size: Batch size for training
        
    Returns:
        train_data: Training data generator
        test_data: Testing data generator
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )
    
    # Test data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directories
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_data, test_data


def preprocess_single_image(img_path, img_size=(100, 100)):
    """
    Preprocess a single image for prediction
    
    Args:
        img_path: Path to the image file
        img_size: Tuple of (height, width) for resizing
        
    Returns:
        Preprocessed image array ready for prediction
    """
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def get_class_names(data_generator):
    """
    Extract class names from data generator
    
    Args:
        data_generator: Keras ImageDataGenerator flow object
        
    Returns:
        List of class names
    """
    class_indices = data_generator.class_indices
    class_names = list(class_indices.keys())
    return class_names


def save_uploaded_images(uploaded_files, save_dir):
    """
    Save uploaded images to a directory for retraining
    
    Args:
        uploaded_files: List of uploaded file objects
        save_dir: Directory to save the images
        
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    
    for file in uploaded_files:
        file_path = os.path.join(save_dir, file.filename)
        file.save(file_path)
        saved_paths.append(file_path)
    
    return saved_paths