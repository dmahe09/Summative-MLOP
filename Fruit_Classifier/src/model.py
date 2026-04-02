"""
Model creation and training module for fruit classification
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

def create_cnn_model(input_shape, num_classes, learning_rate=0.001):
    """
    Create an optimized CNN model for fruit classification.
    
    Optimization techniques used:
    - Batch Normalization for faster convergence
    - L2 Regularization to prevent overfitting
    - Adam optimizer with custom learning rate
    - Dropout for regularization
    
    Args:
        input_shape: Tuple of input image dimensions (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Use Adam optimizer with custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, train_data, test_data, epochs=20, model_save_path='../models/fruit_classifier.h5'):
    """
    Train the model with callbacks for optimization
    
    Args:
        model: Compiled Keras model
        train_data: Training data generator
        test_data: Validation/test data generator
        epochs: Number of training epochs
        model_save_path: Path to save the trained model
        
    Returns:
        Training history object
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    print(f"Model saved to {model_save_path}")
    return history


def retrain_model(existing_model_path, train_data, test_data, epochs=10):
    """
    Retrain an existing model with new data
    
    Args:
        existing_model_path: Path to the existing trained model
        train_data: New training data generator
        test_data: Validation data generator
        epochs: Number of retraining epochs
        
    Returns:
        Retrained model and history
    """
    # Load existing model
    model = tf.keras.models.load_model(existing_model_path)
    print(f"Loaded model from {existing_model_path}")
    
    # Retrain with new data
    history = train_model(model, train_data, test_data, epochs, existing_model_path)
    
    return model, history