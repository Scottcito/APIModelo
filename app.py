from flask import Flask, request, jsonify
import boto3
import torch
from io import BytesIO
from PIL import Image
import os
import torchvision.transforms as transforms

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
        
        # Cargar el modelo
        model = torch.load(model_file, map_location=torch.device('cpu'))
        model.eval()
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
        img = Image.open(file.stream).convert('RGB')  # Convertir a RGB para asegurar compatibilidad
        
        # Convertir la imagen a tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Redimensionar si es necesario
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0)  # Añadir batch dimension
        
        # Realizar la inferencia
        with torch.no_grad():
            results = model(img_tensor)
        
        # Procesar los resultados (ajusta según cómo devuelva tu modelo)
        labels = [result['label'] for result in results]
        
        return jsonify({"labels": labels})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
