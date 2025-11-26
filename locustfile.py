from locust import HttpUser, task, between
import random
import json

class CropDiseaseUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_healthy(self):
        """Simulate prediction request with healthy plant image"""
        files = self._get_random_image('healthy')
        with open(files['file'], 'rb') as f:
            self.client.post("/predict", files={"file": f})
    
    @task(2)
    def predict_diseased(self):
        """Simulate prediction request with diseased plant image"""
        files = self._get_random_image('diseased')
        with open(files['file'], 'rb') as f:
            self.client.post("/predict", files={"file": f})
    
    @task(1)
    def health_check(self):
        """Check API health"""
        self.client.get("/health")
    
    @task(1)
    def model_info(self):
        """Get model information"""
        self.client.get("/model_info")
    
    def _get_random_image(self, category='healthy'):
        """Get random test image path"""
        test_images = {
            'healthy': [
                'test_images/healthy_tomato.jpg',
                'test_images/healthy_potato.jpg'
            ],
            'diseased': [
                'test_images/diseased_tomato.jpg',
                'test_images/diseased_potato.jpg'
            ]
        }
        return {'file': random.choice(test_images[category])}