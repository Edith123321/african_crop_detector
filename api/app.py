from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropDiseaseAPI:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Load the trained model and configuration"""
        try:
            # Load deployment config
            with open('models/deployment_config.json', 'r') as f:
                config = json.load(f)
            
            self.class_names = config['class_names']
            self.model = tf.keras.models.load_model(config['model_path'])
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        image = image.resize(self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array
    
    def predict(self, image):
        """Make prediction on image"""
        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(np.max(predictions[0]))
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(predictions[0][idx])
                }
                for idx in top_3_indices
            ]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}

# Initialize API
api = CropDiseaseAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop disease from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Make prediction
        result = api.predict(image)
        
        if 'error' in result:
            return jsonify(result), 500
        
        logger.info(f"Prediction: {result['predicted_class']} with {result['confidence']:.3f} confidence")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'class_names': api.class_names,
        'input_size': api.img_size,
        'total_classes': len(api.class_names)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)