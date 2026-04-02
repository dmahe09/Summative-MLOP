"""
Streamlit UI for Fruit Classifier - Alternative Demo Interface
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import os
import sys

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_PATH = 'models/fruit_classifier.h5'
CLASS_NAMES = ['apple', 'avocado', 'banana', 'cucumber', 'eggplant', 'mango', 'onion', 'orange']
IMG_SIZE = (100, 100)

# Initialize session state
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'is_retraining' not in st.session_state:
    st.session_state.is_retraining = False
if 'retrain_status' not in st.session_state:
    st.session_state.retrain_status = ""

@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Resize
    image = image.resize(IMG_SIZE)
    # Convert to array
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fruit(model, image):
    """Make prediction on uploaded image"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    start = time.time()
    prediction = model.predict(processed_img, verbose=0)
    latency = (time.time() - start) * 1000
    
    # Get results
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    
    return class_idx, confidence, prediction[0], latency

def save_uploaded_files(uploaded_files, class_name):
    """Save uploaded files for retraining"""
    # Create directory structure
    retrain_dir = os.path.join('retrain_data', class_name)
    os.makedirs(retrain_dir, exist_ok=True)
    
    saved_files = []
    for uploaded_file in uploaded_files:
        # Save file
        file_path = os.path.join(retrain_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    
    return saved_files

def retrain_model_function():
    """Retrain the model with new data"""
    try:
        st.session_state.is_retraining = True
        st.session_state.retrain_status = "Loading existing model..."
        
        # Import necessary modules
        from src.preprocessing import create_data_generators
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        import shutil
        
        # Load existing model
        model = load_model(MODEL_PATH)
        
        # Recompile the model to ensure it's properly configured
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.session_state.retrain_status = "Model loaded. Preparing data..."
        
        # Check if retrain_data exists and has data
        if not os.path.exists('retrain_data') or not os.listdir('retrain_data'):
            st.session_state.retrain_status = "Error: No training data found in retrain_data folder. Please upload images first."
            st.session_state.is_retraining = False
            return False, None
        
        # Copy new data from retrain_data to data/training
        st.session_state.retrain_status = "Merging new training data with existing dataset..."
        
        for class_folder in os.listdir('retrain_data'):
            src_class_path = os.path.join('retrain_data', class_folder)
            dest_class_path = os.path.join('data/training', class_folder)
            
            # Create class folder in training if it doesn't exist
            os.makedirs(dest_class_path, exist_ok=True)
            
            # Copy images
            if os.path.isdir(src_class_path):
                for img_file in os.listdir(src_class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_file = os.path.join(src_class_path, img_file)
                        dest_file = os.path.join(dest_class_path, f"retrain_{img_file}")
                        
                        # Copy if not already exists
                        if not os.path.exists(dest_file):
                            shutil.copy2(src_file, dest_file)
        
        st.session_state.retrain_status = "Data merged. Creating data generators..."
        
        # Create data generators with merged data
        train_data, test_data = create_data_generators(
            'data/training',
            'data/test',
            img_size=(100, 100),
            batch_size=32
        )
        
        st.session_state.retrain_status = "Data prepared. Starting retraining..."
        
        # Define callbacks
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
        
        # Retrain with progress bar
        progress_placeholder = st.empty()
        
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_placeholder.progress((epoch + 1) / 5, text=f"Epoch {epoch + 1}/5")
        
        # Retrain
        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=5,  # Fewer epochs for retraining
            callbacks=[early_stopping, reduce_lr, StreamlitCallback()],
            verbose=1
        )
        
        # Save model
        model.save(MODEL_PATH)
        
        st.session_state.retrain_status = "Retraining complete!"
        st.session_state.is_retraining = False
        
        # Clear model cache to load new version
        st.cache_resource.clear()
        
        return True, history
        
    except Exception as e:
        st.session_state.is_retraining = False
        st.session_state.retrain_status = f"Error: {str(e)}"
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return False, None

# Main UI
st.title("🍎 Fruit Classifier MLOps Dashboard")
st.markdown("### AI-Powered Fruit Recognition System")

# Create tabs
tab1, tab2 = st.tabs(["🔮 Prediction", "🔄 Retrain Model"])

# Sidebar
with st.sidebar:
    st.header("📊 System Metrics")
    
    # Uptime
    uptime = datetime.now() - st.session_state.start_time
    hours = uptime.total_seconds() // 3600
    minutes = (uptime.total_seconds() % 3600) // 60
    st.metric("⏱️ Uptime", f"{int(hours)}h {int(minutes)}m")
    
    # Prediction count
    st.metric("🔮 Total Predictions", st.session_state.prediction_count)
    
    # Model info
    st.metric("🤖 Model Version", "1.0")
    
    # Retraining status
    if st.session_state.is_retraining:
        st.warning("⚠️ Retraining in progress...")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.info("""
    This application uses a Convolutional Neural Network to classify 8 different types of fruits:
    
    🍎 Apple | 🥑 Avocado | 🍌 Banana | 🥒 Cucumber  
    🍆 Eggplant | 🥭 Mango | 🧅 Onion | 🍊 Orange
    """)
    
    st.divider()
    
    st.header("📈 Model Stats")
    st.markdown("""
    - **Accuracy:** 98.7%
    - **Classes:** 8
    - **Framework:** TensorFlow
    - **Architecture:** CNN
    """)

# TAB 1: PREDICTION
with tab1:
    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🔮 Make a Prediction")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a fruit image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a fruit to classify",
            key="predict_uploader"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🚀 Classify Fruit", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model = load_trained_model()
                    
                    if model is not None:
                        # Make prediction
                        class_idx, confidence, all_preds, latency = predict_fruit(model, image)
                        
                        # Update counter
                        st.session_state.prediction_count += 1
                        
                        # Display in col2
                        with col2:
                            st.success("✅ Classification Complete!")
                            
                            # Main prediction
                            st.header("📊 Results")
                            predicted_class = CLASS_NAMES[class_idx].upper()
                            st.markdown(f"## 🎯 Predicted: **{predicted_class}**")
                            
                            # Confidence
                            st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
                            
                            # Latency
                            st.metric("⚡ Response Time", f"{latency:.2f} ms")
                            
                            st.divider()
                            
                            # All predictions
                            st.subheader("📈 All Class Probabilities")
                            
                            # Create dataframe for better display
                            import pandas as pd
                            df = pd.DataFrame({
                                'Fruit': [name.capitalize() for name in CLASS_NAMES],
                                'Confidence': [f"{pred:.1%}" for pred in all_preds],
                                'Score': all_preds
                            })
                            df = df.sort_values('Score', ascending=False)
                            
                            # Display as bar chart
                            st.bar_chart(df.set_index('Fruit')['Score'])
                            
                            # Display as table
                            st.dataframe(
                                df[['Fruit', 'Confidence']],
                                hide_index=True,
                                use_container_width=True
                            )

    with col2:
        if uploaded_file is None:
            st.header("👋 Welcome!")
            st.info("""
            ### How to use:
            
            1. **Upload an image** of a fruit using the uploader on the left
            2. Click **"Classify Fruit"** to get predictions
            3. View the **results** with confidence scores
            
            ### Supported Fruits:
            - 🍎 Apple
            - 🥑 Avocado
            - 🍌 Banana
            - 🥒 Cucumber
            - 🍆 Eggplant
            - 🥭 Mango
            - 🧅 Onion
            - 🍊 Orange
            
            ### Tips:
            - Use clear, well-lit images
            - Single fruit per image works best
            - JPG, JPEG, or PNG formats
            """)

# TAB 2: RETRAIN MODEL
with tab2:
    st.header("🔄 Retrain Model")
    st.markdown("Upload new training data and retrain the model to improve its performance.")
    
    # Create two columns for upload section
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        st.subheader("📁 Upload Training Data")
        
        # Class name input
        class_name = st.text_input(
            "Class Name",
            placeholder="e.g., apple, banana, mango",
            help="Enter the fruit class name for the uploaded images"
        )
        
        # Multiple file uploader
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Select multiple images of the same fruit class",
            key="retrain_uploader"
        )
        
        # Display uploaded files
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} images ready to upload")
            
            # Show preview of first few images
            with st.expander("👁️ Preview Images"):
                cols = st.columns(4)
                for idx, file in enumerate(uploaded_files[:8]):
                    with cols[idx % 4]:
                        image = Image.open(file)
                        st.image(image, caption=f"Image {idx+1}", use_container_width=True)
        
        # Upload button
        if st.button("📤 Upload Training Data", type="secondary", use_container_width=True, disabled=st.session_state.is_retraining):
            if not class_name:
                st.error("❌ Please enter a class name!")
            elif not uploaded_files:
                st.error("❌ Please upload at least one image!")
            else:
                with st.spinner(f"Uploading {len(uploaded_files)} images..."):
                    saved_files = save_uploaded_files(uploaded_files, class_name.lower())
                    st.success(f"✅ Successfully uploaded {len(saved_files)} images to class '{class_name}'!")
                    st.balloons()
    
    with upload_col2:
        st.subheader("📊 Upload Stats")
        
        # Count files in retrain_data
        retrain_dir = 'retrain_data'
        if os.path.exists(retrain_dir):
            total_files = 0
            classes = {}
            for class_folder in os.listdir(retrain_dir):
                class_path = os.path.join(retrain_dir, class_folder)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    classes[class_folder] = count
                    total_files += count
            
            st.metric("Total Images", total_files)
            st.metric("Classes", len(classes))
            
            if classes:
                st.markdown("**Per Class:**")
                for cls, cnt in classes.items():
                    st.markdown(f"- {cls}: {cnt} images")
        else:
            st.info("No training data uploaded yet")
    
    st.divider()
    
    # Retrain section
    retrain_col1, retrain_col2 = st.columns([3, 1])
    
    with retrain_col1:
        st.subheader("🚀 Start Retraining")
        st.markdown("""
        **Before retraining:**
        - Ensure you have uploaded sufficient training data (recommended: 50+ images per class)
        - Retraining will take several minutes depending on data size
        - The model will be automatically saved after successful training
        """)
        
        # Status message
        if st.session_state.retrain_status:
            if "complete" in st.session_state.retrain_status.lower():
                st.success(st.session_state.retrain_status)
            elif "error" in st.session_state.retrain_status.lower():
                st.error(st.session_state.retrain_status)
            else:
                st.info(st.session_state.retrain_status)
    
    with retrain_col2:
        # Retrain button
        if st.button(
            "🔄 Start Retraining",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_retraining,
            help="Click to start model retraining with uploaded data"
        ):
            if st.session_state.is_retraining:
                st.warning("⚠️ Retraining already in progress!")
            else:
                with st.spinner("🔄 Retraining model... This may take several minutes."):
                    success, history = retrain_model_function()
                    
                    if success:
                        st.success("✅ Retraining completed successfully!")
                        st.balloons()
                        
                        # Show training history
                        if history:
                            st.subheader("📈 Training History")
                            
                            import pandas as pd
                            history_df = pd.DataFrame({
                                'Epoch': range(1, len(history.history['accuracy']) + 1),
                                'Training Accuracy': history.history['accuracy'],
                                'Validation Accuracy': history.history['val_accuracy']
                            })
                            
                            st.line_chart(history_df.set_index('Epoch'))
                    else:
                        st.error("❌ Retraining failed. Check the error message above.")
        
        # Clear cache button
        if st.button("🔄 Reload Model", use_container_width=True, help="Reload the model from disk"):
            st.cache_resource.clear()
            st.success("Model cache cleared! Model will reload on next prediction.")
    
    st.divider()
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        st.markdown("""
        ### Training Configuration
        - **Epochs:** 5 (with early stopping)
        - **Batch Size:** 32
        - **Optimizer:** Adam
        - **Learning Rate:** 0.001 (with reduction on plateau)
        - **Callbacks:** Early Stopping, Learning Rate Reduction
        
        ### Data Augmentation
        - Rotation: ±20°
        - Zoom: ±20%
        - Horizontal Flip: Random
        
        ### Model Architecture
        - 3 Convolutional Blocks
        - Batch Normalization
        - L2 Regularization (0.001)
        - Dropout (0.25, 0.5)
        """)

# Footer
st.divider()
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**🎓 MLOps Pipeline Project**")
with col_b:
    st.markdown("**🔧 TensorFlow + Streamlit**")
with col_c:
    st.markdown("**📊 Real-time Classification**")

# Additional features in expander
with st.expander("🔬 Technical Details"):
    st.markdown("""
    ### Model Architecture
    - **Input:** 100x100x3 RGB images
    - **Layers:** 3 Conv blocks + 2 Dense layers
    - **Optimization:** BatchNorm + L2 Regularization + Dropout
    - **Optimizer:** Adam (lr=0.001)
    - **Training:** Early stopping + LR reduction
    
    ### Performance Metrics
    - **Validation Accuracy:** 98.7%
    - **Macro F1-Score:** 98.5%
    - **Average Inference Time:** ~650ms
    - **Model Size:** ~15 MB
    
    ### Dataset
    - **Training Samples:** 3,937 images
    - **Test Samples:** 1,312 images
    - **Source:** Fruits-360 Dataset
    """)

with st.expander("📁 Sample Images"):
    st.markdown("""
    ### Example Test Images
    You can find sample images in:
    - `data/test/apple/`
    - `data/test/banana/`
    - `data/test/orange/`
    - etc.
    
    Try uploading images from these folders!
    """)