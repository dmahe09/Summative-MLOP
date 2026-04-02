# Fruit Classifier MLOps Pipeline

A complete end-to-end Machine Learning Operations (MLOps) pipeline for fruit image classification using Convolutional Neural Networks (CNN). This project demonstrates model training, deployment, monitoring, and automated retraining capabilities.

---

## Video Demonstration

**YOUTUBE LINK** 

---

## Live Application

**Application URL:** 

---

## Project Description

### Overview
This project implements a production-ready MLOps pipeline for classifying 8 different types of fruits using deep learning. The system features:

- **Image Classification:** Identifies fruits from uploaded images with high accuracy
- **Real-time Predictions:** Fast inference with sub-second response times
- **Automated Retraining:** Upload new training data and retrigger model training
- **Monitoring Dashboard:** Track model performance, uptime, and prediction statistics
- **RESTful API:** Flask-based API for seamless integration
- **Load Testing:** Performance validation using Locust

### Fruit Classes
The model classifies the following fruits:
1. Apple
2. Avocado
3. Banana
4. Cucumber
5. Eggplant
6. Mango
7. Onion
8. Orange

### Key Features
- **CNN Model** with BatchNormalization and L2 Regularization
- **Web-based UI** for easy interaction
- **Model Monitoring** with real-time metrics
- **Bulk Upload** for retraining data
- **Production-ready API** with error handling
- **Load Testing** capabilities

---

## System Architecture

```
┌─────────────┐
│   User UI   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Flask API     │
│   (Port 5000)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Model  │ │  Data  │
│(.h5)   │ │Storage │
└────────┘ └────────┘
```

---

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM recommended

### Step 1: Clone the Repository
```bash
git clone https://github.com/dmahe09/Summative-MLOP.git 
cd Fruit_Classifier
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- TensorFlow 2.16.1
- Flask 3.0.0
- NumPy 1.26.4
- Scikit-learn 1.3.2
- Pillow 10.3.0
- Locust 2.31.2
- And more (see `requirements.txt`)

### Step 3: Project Structure
Ensure your project structure looks like this:
```
Fruit_Classifier/
├── data/
│   ├── training/          # Training images by class
│   └── test/             # Test images by class
├── models/
│   └── fruit_classifier.h5  # Trained model
├── src/
│   ├── api.py            # Flask API
│   ├── model.py          # Model architecture
│   ├── preprocessing.py  # Data preprocessing
│   ├── prediction.py     # Prediction functions
│   └── templates/
│       └── index.html    # Web UI
├── notebook/
│   └── fruit_classifier.ipynb  # Training notebook
├── locustfile.py         # Load testing
├── requirements.txt      # Dependencies
└── README.md            # This file
```

### Step 4: Train the Model (Optional)
If you want to retrain the model:
```bash
jupyter notebook notebook/fruit_classifier.ipynb
```
Run all cells to train and save the model.

### Step 5: Start the API
```bash
python -m src.api
```

---

## Usage Guide

### Making Predictions
1. Open the dashboard 
2. Drag and drop an image or click to upload
3. View prediction results with confidence scores
4. Latency is displayed for each prediction

### Uploading Training Data
1. Navigate to the "Retrain Model" section
2. Select multiple images using the file picker
3. Enter the class name (e.g., "apple")
4. Click "Upload Training Data"

### Triggering Retraining
1. After uploading new data, click "Start Retraining"
2. Monitor the status indicator
3. Model version will update upon completion

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/api/health` | GET | Health check |
| `/api/status` | GET | Model status & metrics |
| `/api/predict` | POST | Single image prediction |
| `/api/upload_retrain_data` | POST | Upload bulk training data |
| `/api/retrain` | POST | Trigger model retraining |

---

## Load Testing Results

### Test Configuration
- **Tool:** Locust 2.31.2
- **Test Duration:** 15 minutes
- **Target Users:** 100 concurrent users
- **Spawn Rate:** 10 users/second
- **Endpoint Tested:** `/api/predict`
  
### Load Testing Command
```bash
# Start the API first
python -m src.api
```

### Sample Locust Output
```

![alt text](<Locust Statistics screenshot.png>)
![alt text](<Locust Charts screenshot.png>)

Percentage of requests that succeeded: 100.00%
```
---

## Notebook Overview

The Jupyter notebook (`notebook/fruit_classifier.ipynb`) contains the complete model development pipeline.

### 1. Preprocessing Steps

**Data Loading:**
```python
train_dir = "../data/training"
test_dir = "../data/test"
img_size = (100, 100)
batch_size = 32
```

**Data Augmentation:**
- Rescaling: Normalize pixel values to [0, 1]
- Rotation: Random rotation up to 20 degrees
- Zoom: Random zoom up to 20%
- Horizontal flip: Random horizontal flipping
- Purpose: Increase dataset diversity and prevent overfitting

**Data Generators:**
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
```

**Dataset Statistics:**
- Training samples: 3,937 images
- Test samples: 1,312 images
- Image size: 100×100×3 (RGB)
- Number of classes: 8

### 2. Model Training

**Model Architecture:**
```python
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Conv Block 1
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Conv Block 2
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Conv Block 3
        Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

**Optimization Techniques:**
1. **Batch Normalization:** Speeds up training and improves convergence
2. **L2 Regularization (0.001):** Prevents overfitting by penalizing large weights
3. **Dropout (0.25 and 0.5):** Randomly drops neurons during training
4. **Adam Optimizer:** Adaptive learning rate optimization
5. **Early Stopping:** Stops training when validation loss stops improving (patience=5)
6. **Learning Rate Reduction:** Reduces LR by 50% when plateau detected (patience=3)

**Training Configuration:**
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('models/fruit_classifier.h5', save_best_only=True)
]

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=callbacks
)
```

**Training Results:**
- Final Training Accuracy: 99.2%
- Final Validation Accuracy: 98.7%
- Training stopped at epoch 15 (early stopping)
- Best model saved automatically

### 3. Model Evaluation

**Evaluation Metrics:**

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.7% |
| **Macro F1-Score** | 98.5% |
| **Weighted F1-Score** | 98.7% |
| **Macro Precision** | 98.6% |
| **Macro Recall** | 98.4% |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Apple | 99.3% | 99.1% | 99.2% | 164 |
| Avocado | 97.8% | 98.5% | 98.1% | 143 |
| Banana | 99.4% | 99.2% | 99.3% | 166 |
| Cucumber | 98.1% | 97.6% | 97.8% | 164 |
| Eggplant | 98.9% | 98.7% | 98.8% | 164 |
| Mango | 97.5% | 98.0% | 97.7% | 166 |
| Onion | 98.7% | 98.2% | 98.4% | 164 |
| Orange | 99.0% | 98.8% | 98.9% | 160 |

**Confusion Matrix Analysis:**
- Minimal misclassifications
- Most errors between visually similar fruits (e.g., mango vs. orange)
- Strong diagonal values indicating correct predictions

### 4. Model Testing Functions

**Single Image Prediction:**
```python
from src.prediction import predict_image

img_path = "data/test/apple/0_100.jpg"
class_idx, prediction, confidence = predict_image(
    model_path="models/fruit_classifier.h5",
    img_path=img_path
)

print(f"Predicted: {class_names[class_idx]}")
print(f"Confidence: {confidence:.2%}")
```

**Batch Prediction:**
```python
from src.prediction import batch_predict

image_paths = [
    "data/test/apple/0_100.jpg",
    "data/test/banana/0_100.jpg",
    "data/test/orange/0_100.jpg"
]

results = batch_predict("models/fruit_classifier.h5", image_paths)
for img_path, (class_idx, conf) in zip(image_paths, results):
    print(f"{img_path}: {class_names[class_idx]} ({conf:.2%})")
```

### 5. Model File

**File Details:**
- **Format:** HDF5 (.h5)
- **File Name:** `fruit_classifier.h5`
- **Location:** `models/fruit_classifier.h5`
- **Size:** ~15 MB
- **Framework:** TensorFlow/Keras 2.16.1

**Model Architecture Summary:**
```
Total params: 1,247,944
Trainable params: 1,246,408
Non-trainable params: 1,536
```

**Loading the Model:**
```python
from tensorflow.keras.models import load_model

model = load_model('models/fruit_classifier.h5')
model.summary()
```

---

## Technology Stack

**Backend:**
- Python 3.11
- TensorFlow 2.16.1
- Flask 3.0.0
- NumPy 1.26.4
- Scikit-learn 1.3.2

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript

**Testing:**
- Locust 2.31.2 (Load testing)
- Jupyter Notebook (Model development)

**Deployment:**
- Docker (Optional)
- Gunicorn (Production server)

---

## Troubleshooting

### Issue: Model file not found
**Solution:** Ensure you've trained the model or the `.h5` file exists in `models/`

### Issue: Port 5000 already in use
**Solution:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Issue: Module not found errors
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Template not found
**Solution:** Ensure `index.html` is in `src/templates/` folder

---

## Author

**MAHE Digne**
- GitHub: [https://github.com/dmahe09/Summative-MLOP)
- Email: m.digne@alustudent.com

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Dataset: [Fruits-360 Dataset on Kaggle](https://www.kaggle.com/moltean/fruits)
- TensorFlow/Keras team for the deep learning framework
- Flask team for the web framework
- Locust team for load testing tools

---