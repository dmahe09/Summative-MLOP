"""
Standalone retraining script for Fruit Classifier
Can be run manually or triggered by API
"""
import os
import sys
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import create_data_generators
from src.model import create_cnn_model


def retrain_model(
    existing_model_path='models/fruit_classifier.h5',
    train_dir='data/training',
    test_dir='data/test',
    epochs=10,
    img_size=(100, 100),
    batch_size=32,
    save_path=None
):
    """
    Retrain existing model with new data
    
    Args:
        existing_model_path: Path to existing trained model
        train_dir: Directory containing training data
        test_dir: Directory containing validation data
        epochs: Number of epochs to retrain
        img_size: Image dimensions (height, width)
        batch_size: Batch size for training
        save_path: Path to save retrained model (defaults to existing_model_path)
    
    Returns:
        model: Retrained model
        history: Training history
    """
    
    print("="*60)
    print("RETRAINING MODEL")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    print(f"Existing model: {existing_model_path}")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {test_dir}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    
    # Set save path
    if save_path is None:
        save_path = existing_model_path
    
    # Check if model exists
    if not os.path.exists(existing_model_path):
        print(f"❌ Error: Model not found at {existing_model_path}")
        print("Creating new model instead...")
        
        # Create data generators to get number of classes
        train_data, _ = create_data_generators(train_dir, test_dir, img_size, batch_size)
        num_classes = len(train_data.class_indices)
        
        # Create new model
        model = create_cnn_model(
            input_shape=(img_size[0], img_size[1], 3),
            num_classes=num_classes
        )
    else:
        # Load existing model
        print(f"✅ Loading existing model from {existing_model_path}")
        model = load_model(existing_model_path)
        print(f"Model loaded successfully!")
        print(f"Model has {model.count_params():,} parameters")
    
    # Create data generators
    print("\n📊 Creating data generators...")
    train_data, test_data = create_data_generators(
        train_dir, 
        test_dir, 
        img_size, 
        batch_size
    )
    
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {test_data.samples}")
    print(f"Number of classes: {len(train_data.class_indices)}")
    print(f"Classes: {list(train_data.class_indices.keys())}")
    
    # Define callbacks
    print("\n⚙️ Setting up callbacks...")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model_checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
    
    # Retrain model
    print("\n🚀 Starting retraining...")
    print("-"*60)
    
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final model
    print("\n📈 Evaluating retrained model...")
    final_loss, final_accuracy = model.evaluate(test_data, verbose=0)
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now()}")
    print(f"Final validation loss: {final_loss:.4f}")
    print(f"Final validation accuracy: {final_accuracy:.4f}")
    print(f"Model saved to: {save_path}")
    print("="*60)
    
    return model, history


def retrain_with_new_data(
    existing_model_path='models/fruit_classifier.h5',
    new_data_dir='retrain_data',
    original_train_dir='data/training',
    test_dir='data/test',
    epochs=10
):
    """
    Retrain model incorporating new uploaded data
    
    Args:
        existing_model_path: Path to existing model
        new_data_dir: Directory containing newly uploaded data
        original_train_dir: Original training data directory
        test_dir: Test/validation directory
        epochs: Number of epochs to retrain
    
    Returns:
        model: Retrained model
        history: Training history
    """
    
    print("\n🔄 Retraining with new data...")
    print(f"New data directory: {new_data_dir}")
    
    # Check if new data exists
    if not os.path.exists(new_data_dir):
        print(f"⚠️ Warning: New data directory {new_data_dir} not found")
        print("Proceeding with original training data only...")
        return retrain_model(
            existing_model_path,
            original_train_dir,
            test_dir,
            epochs
        )
    
    # Count new data
    new_data_count = 0
    for class_folder in os.listdir(new_data_dir):
        class_path = os.path.join(new_data_dir, class_folder)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            new_data_count += len(files)
            print(f"  {class_folder}: {len(files)} new images")
    
    print(f"\nTotal new images: {new_data_count}")
    
    if new_data_count == 0:
        print("⚠️ No new images found. Using original training data only...")
        return retrain_model(
            existing_model_path,
            original_train_dir,
            test_dir,
            epochs
        )
    
    # TODO: Optionally merge new_data_dir with original_train_dir
    # For now, we'll use the new data directory if it has data
    
    # Retrain with new data
    return retrain_model(
        existing_model_path,
        new_data_dir,  # Use new data for retraining
        test_dir,
        epochs
    )


def main():
    """Command line interface for retraining"""
    parser = argparse.ArgumentParser(description='Retrain Fruit Classifier Model')
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/fruit_classifier.h5',
        help='Path to existing model'
    )
    
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/training',
        help='Training data directory'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='data/test',
        help='Test/validation data directory'
    )
    
    parser.add_argument(
        '--new-data-dir',
        type=str,
        default=None,
        help='Directory with newly uploaded data'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs to retrain'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=100,
        help='Image size (height and width)'
    )
    
    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='Path to save retrained model (default: overwrites existing)'
    )
    
    args = parser.parse_args()
    
    # If new data directory is specified, use it
    if args.new_data_dir:
        model, history = retrain_with_new_data(
            existing_model_path=args.model,
            new_data_dir=args.new_data_dir,
            original_train_dir=args.train_dir,
            test_dir=args.test_dir,
            epochs=args.epochs
        )
    else:
        # Standard retraining
        model, history = retrain_model(
            existing_model_path=args.model,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            epochs=args.epochs,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            save_path=args.save_path
        )
    
    print("\n✅ Retraining script completed successfully!")


if __name__ == '__main__':
    main()