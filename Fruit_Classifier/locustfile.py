"""
Locust load testing file for Fruit Classifier API
Run with: locust -f locustfile.py --host=http://localhost:5000
"""
from locust import HttpUser, task, between
import os
import random

class FruitClassifierUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Get list of test images
        self.test_images = []
        test_dir = "data/test"
        
        if os.path.exists(test_dir):
            for class_folder in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_folder)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path)[:5]:  # Take 5 images per class
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.test_images.append(os.path.join(class_path, img_file))
    
    @task(3)  # Weight: 3x more likely than other tasks
    def predict_image(self):
        """Test the prediction endpoint"""
        if not self.test_images:
            return
        
        # Select random image
        img_path = random.choice(self.test_images)
        
        with open(img_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            with self.client.post('/api/predict', files=files, catch_response=True) as response:
                if response.status_code == 200:
                    json_data = response.json()
                    if json_data.get('success'):
                        response.success()
                    else:
                        response.failure(f"Prediction failed: {json_data.get('error')}")
                else:
                    response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_status(self):
        """Test the status endpoint"""
        with self.client.get('/api/status', catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test the health check endpoint"""
        with self.client.get('/api/health', catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class StressTestUser(HttpUser):
    """Aggressive stress testing user"""
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    def on_start(self):
        self.test_images = []
        test_dir = "data/test"
        
        if os.path.exists(test_dir):
            for class_folder in os.listdir(test_dir):
                class_path = os.path.join(test_dir, class_folder)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path)[:3]:
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.test_images.append(os.path.join(class_path, img_file))
    
    @task
    def rapid_fire_predictions(self):
        """Rapid fire prediction requests"""
        if not self.test_images:
            return
        
        img_path = random.choice(self.test_images)
        
        with open(img_path, 'rb') as f:
            files = {'file': ('stress_test.jpg', f, 'image/jpeg')}
            self.client.post('/api/predict', files=files)