from locust import HttpUser, task, between, TaskSet
import os
import random
import time
from pathlib import Path

class ImageLoader:
    """Helper class to load test images from all directories"""
    
    def __init__(self, base_test_path="data/test"):
        self.base_test_path = base_test_path
        self.all_images = self._load_all_images()
        self.by_category = self._categorize_images()
    
    def _load_all_images(self):
        """Load all image paths from the test directory structure"""
        all_images = []
        base_path = Path(self.base_test_path)
        
        if not base_path.exists():
            raise Exception(f"Test data directory not found: {base_path}")
        
        # Walk through all subdirectories
        for category_dir in base_path.iterdir():
            if category_dir.is_dir():
                # Get all image files in this category
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(category_dir.glob(ext))
                
                for img_path in image_files:
                    all_images.append({
                        'path': str(img_path),
                        'category': category_dir.name,
                        'filename': img_path.name
                    })
        
        if not all_images:
            raise Exception(f"No test images found in {base_path}")
        
        print(f"Loaded {len(all_images)} test images from {len(list(base_path.iterdir()))} categories")
        return all_images
    
    def _categorize_images(self):
        """Categorize images by their folder names"""
        categorized = {}
        for img in self.all_images:
            category = img['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(img)
        return categorized
    
    def get_random_image(self):
        """Get a random image from any category"""
        return random.choice(self.all_images)
    
    def get_image_by_category(self, category):
        """Get a random image from a specific category"""
        if category in self.by_category and self.by_category[category]:
            return random.choice(self.by_category[category])
        return None
    
    def get_categories(self):
        """Get list of all available categories"""
        return list(self.by_category.keys())

class NavigationTasks(TaskSet):
    """Task set for navigation-related tests"""
    
    @task(3)
    def load_main_page(self):
        with self.client.get("/", catch_response=True, name="Navigation - Main Page") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to load main page: {response.status_code}")
    
    @task(2)
    def check_dashboard_metrics(self):
        with self.client.get("/", catch_response=True, name="Navigation - Dashboard Metrics") as response:
            if response.status_code == 200:
                # Check for various dashboard elements
                checks = [
                    "accuracy" in response.text.lower(),
                    "model" in response.text.lower(),
                    "system" in response.text.lower()
                ]
                if any(checks):
                    response.success()
                else:
                    response.failure("Dashboard metrics not found")
            else:
                response.failure(f"Dashboard check failed: {response.status_code}")
    
    @task(1)
    def check_disease_categories(self):
        with self.client.get("/", catch_response=True, name="Navigation - Disease Categories") as response:
            if response.status_code == 200:
                # Check for disease category mentions
                disease_indicators = [
                    "tomato" in response.text.lower(),
                    "potato" in response.text.lower(), 
                    "pepper" in response.text.lower(),
                    "healthy" in response.text.lower()
                ]
                if any(disease_indicators):
                    response.success()
                else:
                    response.failure("Disease categories not found")
            else:
                response.failure(f"Disease categories check failed: {response.status_code}")

class UploadTasks(TaskSet):
    """Task set for file upload and analysis tests"""
    
    def on_start(self):
        """Initialize image loader when task set starts"""
        self.image_loader = self.user.image_loader
    
    @task(5)
    def upload_random_disease_image(self):
        """Upload a random disease image for analysis"""
        image_info = self.image_loader.get_random_image()
        self._upload_image(image_info, "Random Disease Image")
    
    @task(3)
    def upload_potato_disease(self):
        """Upload a potato disease image specifically"""
        potato_categories = [cat for cat in self.image_loader.get_categories() if "potato" in cat.lower()]
        if potato_categories:
            category = random.choice(potato_categories)
            image_info = self.image_loader.get_image_by_category(category)
            if image_info:
                self._upload_image(image_info, f"Potato Disease - {category}")
            else:
                self._upload_fallback("Potato Disease")
        else:
            self._upload_fallback("Potato Disease")
    
    @task(3)
    def upload_tomato_disease(self):
        """Upload a tomato disease image specifically"""
        tomato_categories = [cat for cat in self.image_loader.get_categories() if "tomato" in cat.lower()]
        if tomato_categories:
            category = random.choice(tomato_categories)
            image_info = self.image_loader.get_image_by_category(category)
            if image_info:
                self._upload_image(image_info, f"Tomato Disease - {category}")
            else:
                self._upload_fallback("Tomato Disease")
        else:
            self._upload_fallback("Tomato Disease")
    
    @task(2)
    def upload_healthy_plant(self):
        """Upload a healthy plant image"""
        healthy_categories = [cat for cat in self.image_loader.get_categories() if "healthy" in cat.lower()]
        if healthy_categories:
            category = random.choice(healthy_categories)
            image_info = self.image_loader.get_image_by_category(category)
            if image_info:
                self._upload_image(image_info, f"Healthy Plant - {category}")
            else:
                self._upload_fallback("Healthy Plant")
        else:
            self._upload_fallback("Healthy Plant")
    
    @task(1)
    def upload_pepper_disease(self):
        """Upload a pepper disease image specifically"""
        pepper_categories = [cat for cat in self.image_loader.get_categories() if "pepper" in cat.lower()]
        if pepper_categories:
            category = random.choice(pepper_categories)
            image_info = self.image_loader.get_image_by_category(category)
            if image_info:
                self._upload_image(image_info, f"Pepper Disease - {category}")
            else:
                self._upload_fallback("Pepper Disease")
        else:
            self._upload_fallback("Pepper Disease")
    
    @task(1)
    def upload_multiple_categories_in_sequence(self):
        """Upload multiple images from different categories in sequence"""
        categories = random.sample(self.image_loader.get_categories(), min(3, len(self.image_loader.get_categories())))
        
        for category in categories:
            image_info = self.image_loader.get_image_by_category(category)
            if image_info:
                self._upload_image(image_info, f"Sequential - {category}")
                # Small delay between sequential uploads
                time.sleep(0.5)
    
    def _upload_image(self, image_info, request_name):
        """Helper method to upload an image"""
        try:
            with open(image_info['path'], 'rb') as image_file:
                files = {
                    'file': (
                        image_info['filename'],
                        image_file,
                        'image/jpeg' if image_info['path'].lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                    )
                }
                
                start_time = time.time()
                
                with self.client.post(
                    "/",
                    files=files,
                    catch_response=True,
                    name=request_name
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        # Check for success indicators in response
                        success_indicators = [
                            "prediction", "confidence", "disease", "healthy",
                            "treatment", "recommendation", "analysis"
                        ]
                        
                        if any(indicator in response.text.lower() for indicator in success_indicators):
                            response.success()
                        else:
                            response.failure("Upload successful but no analysis results detected")
                    else:
                        response.failure(f"Upload failed with status: {response.status_code}")
                        
        except Exception as e:
            self.user.environment.events.request_failure.fire(
                request_type="POST",
                name=request_name,
                response_time=0,
                exception=e,
                response_length=0
            )
    
    def _upload_fallback(self, request_name):
        """Fallback upload if specific category not available"""
        image_info = self.image_loader.get_random_image()
        if image_info:
            self._upload_image(image_info, f"{request_name} - Fallback")

class ComprehensiveCropDiseaseUser(HttpUser):
    """Comprehensive user simulating real usage patterns with all test images"""
    wait_time = between(2, 5)
    
    tasks = {
        NavigationTasks: 2,
        UploadTasks: 8
    }
    
    def on_start(self):
        """Initialize image loader when user starts"""
        try:
            self.image_loader = ImageLoader("data/test")
            print(f"User initialized with {len(self.image_loader.all_images)} test images")
        except Exception as e:
            print(f"Error initializing image loader: {e}")
            raise

class StressTestUser(HttpUser):
    """User for stress testing with minimal wait times"""
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Initialize image loader for stress testing"""
        try:
            self.image_loader = ImageLoader("data/test")
        except Exception as e:
            print(f"Error initializing stress test image loader: {e}")
            raise
    
    @task(6)
    def stress_page_load(self):
        """Repeated page loads for stress testing"""
        self.client.get("/", name="Stress - Page Load")
    
    @task(4)
    def stress_random_upload(self):
        """Repeated random uploads for stress testing"""
        try:
            image_info = self.image_loader.get_random_image()
            with open(image_info['path'], 'rb') as image_file:
                files = {
                    'file': (
                        image_info['filename'],
                        image_file,
                        'image/jpeg' if image_info['path'].lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                    )
                }
                self.client.post("/", files=files, name="Stress - Random Upload")
        except Exception as e:
            # In stress testing, continue even if some operations fail
            pass

class CategorySpecificUser(HttpUser):
    """User that focuses on specific disease categories"""
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize and choose a focus category"""
        self.image_loader = ImageLoader("data/test")
        self.focus_category = random.choice(self.image_loader.get_categories())
        print(f"CategorySpecificUser focusing on: {self.focus_category}")
    
    @task(3)
    def load_main_page(self):
        self.client.get("/", name=f"Category {self.focus_category} - Main Page")
    
    @task(7)
    def upload_focus_category(self):
        """Upload images only from the focus category"""
        image_info = self.image_loader.get_image_by_category(self.focus_category)
        if image_info:
            try:
                with open(image_info['path'], 'rb') as image_file:
                    files = {
                        'file': (
                            image_info['filename'],
                            image_file,
                            'image/jpeg' if image_info['path'].lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                        )
                    }
                    self.client.post("/", files=files, name=f"Focus Category - {self.focus_category}")
            except Exception:
                pass

# Quick test function to verify image loading
def test_image_loading():
    """Test function to verify images are loaded correctly"""
    try:
        loader = ImageLoader("data/test")
        print("✓ Image loading test passed!")
        print(f"✓ Loaded {len(loader.all_images)} images from {len(loader.get_categories())} categories")
        print("✓ Categories:", loader.get_categories())
        
        # Test random image selection
        random_img = loader.get_random_image()
        print(f"✓ Random image: {random_img['category']} - {random_img['filename']}")
        
        return True
    except Exception as e:
        print(f"✗ Image loading test failed: {e}")
        return False

if __name__ == "__main__":
    # Run a quick test when file is executed directly
    test_image_loading()