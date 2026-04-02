"""
Flask API for Fruit Classification MLOps Pipeline
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
from datetime import datetime
import json
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import base64
import threading
import time

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import predict_image
from src.preprocessing import create_data_generators
from src.model import retrain_model

# Get the project root directory (parent of src/)
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask with correct template folder
app = Flask(__name__, 
            template_folder=os.path.join(PROJECT_ROOT, 'templates'),
            static_folder=os.path.join(PROJECT_ROOT, 'static'))
CORS(app)

# Configuration - use absolute paths
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads')
RETRAIN_FOLDER = os.path.join(PROJECT_ROOT, 'retrain_data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'fruit_classifier.h5')
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'training')
TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'test')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_FOLDER, exist_ok=True)

# Global variables for monitoring
model_status = {
    'uptime_start': datetime.now(),
    'total_predictions': 0,
    'last_prediction_time': None,
    'is_retraining': False,
    'last_retrain_time': None,
    'model_version': '1.0'
}

# Class names (update based on your dataset)
CLASS_NAMES = ['apple', 'avocado', 'banana', 'cucumber', 'eggplant', 'mango', 'onion', 'orange']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main UI"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - model_status['uptime_start']).total_seconds()
    return jsonify({
        'status': 'healthy',
        'uptime_seconds': uptime,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model status and metrics"""
    uptime = (datetime.now() - model_status['uptime_start']).total_seconds()
    return jsonify({
        'uptime_seconds': uptime,
        'uptime_formatted': str(datetime.now() - model_status['uptime_start']),
        'total_predictions': model_status['total_predictions'],
        'last_prediction_time': model_status['last_prediction_time'],
        'is_retraining': model_status['is_retraining'],
        'last_retrain_time': model_status['last_retrain_time'],
        'model_version': model_status['model_version']
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict single image"""
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make prediction
            class_idx, prediction, confidence = predict_image(MODEL_PATH, filepath)
            
            # Update metrics
            model_status['total_predictions'] += 1
            model_status['last_prediction_time'] = datetime.now().isoformat()
            
            latency = time.time() - start_time
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predicted_class': CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f'Class_{class_idx}',
                'class_index': int(class_idx),
                'confidence': float(confidence),
                'all_predictions': {CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class_{i}': float(prediction[i]) 
                                   for i in range(len(prediction))},
                'latency_ms': latency * 1000
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/upload_retrain_data', methods=['POST'])
def upload_retrain_data():
    """Upload bulk images for retraining"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    class_name = request.form.get('class_name', 'unknown')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    saved_files = []
    class_folder = os.path.join(RETRAIN_FOLDER, class_name)
    os.makedirs(class_folder, exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(class_folder, filename)
            file.save(filepath)
            saved_files.append(filepath)
    
    return jsonify({
        'success': True,
        'message': f'Uploaded {len(saved_files)} images for class "{class_name}"',
        'saved_count': len(saved_files)
    })


def retrain_model_background():
    """Background task for retraining"""
    try:
        model_status['is_retraining'] = True
        
        # Create data generators with retrain data
        train_data, test_data = create_data_generators(TRAIN_DIR, TEST_DIR)
        
        # Retrain model
        model = tf.keras.models.load_model(MODEL_PATH)
        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=5,  # Fewer epochs for retraining
            verbose=1
        )
        
        # Save retrained model
        model.save(MODEL_PATH)
        
        model_status['is_retraining'] = False
        model_status['last_retrain_time'] = datetime.now().isoformat()
        model_status['model_version'] = f"{float(model_status['model_version']) + 0.1:.1f}"
        
        print("Retraining completed successfully!")
        
    except Exception as e:
        print(f"Retraining failed: {str(e)}")
        model_status['is_retraining'] = False


@app.route('/api/retrain', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining"""
    if model_status['is_retraining']:
        return jsonify({'error': 'Retraining already in progress'}), 400
    
    # Start retraining in background thread
    thread = threading.Thread(target=retrain_model_background)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Retraining started in background',
        'status': 'training'
    })


@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """Get data for visualizations"""
    # This would typically query a database
    # For now, return dummy data
    return jsonify({
        'class_distribution': {
            'apple': 491,
            'banana': 490,
            'orange': 479,
            'mango': 490,
            'avocado': 427,
            'cucumber': 490,
            'eggplant': 490,
            'onion': 490
        },
        'prediction_history': [
            {'timestamp': '2024-01-01T10:00:00', 'class': 'apple', 'confidence': 0.95},
            {'timestamp': '2024-01-01T10:05:00', 'class': 'banana', 'confidence': 0.89},
        ]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)