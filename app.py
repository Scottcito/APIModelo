from flask import Flask, request, jsonify
import boto3
import torch
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import ultralytics
from ultralytics import YOLO
import os

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
        # Descargar el modelo desde S3
        model_file = BytesIO()
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_file)
        model_file.seek(0)

        # Cargar el modelo YOLOv8
        model = YOLO(model_file)  # Usa YOLOv8 de ultralytics
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener el archivo de la solicitud
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')

        # Realizar la inferencia
        results = model(img)  # Inferencia con YOLOv8

        # Procesar los resultados
        predictions = results.pandas().xyxy[0].to_dict(orient='records')  # Convertir a dict

        return jsonify({"predictions": predictions})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
