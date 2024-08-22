from flask import Flask, request, jsonify
import boto3
import torch
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import ultralytics
from ultralytics import YOLO
import os
import tempfile

app = Flask(__name__)

# Configurar el cliente de S3 usando credenciales del entorno
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = 'modeloprueba'
MODEL_KEY = 'abecedariobest.pt'

def load_model():
    try:
        model_file = BytesIO()
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_file)
        model_file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
            temp_model_file.write(model_file.getbuffer())
            temp_model_path = temp_model_file.name

        model = YOLO(temp_model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')

        results = model(img) 

        predictions = results.pandas().xyxy[0].to_dict(orient='records')  # Convertir a dict

        return jsonify({"predictions": predictions})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)