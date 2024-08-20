from flask import Flask, request, jsonify
import boto3
import torch
from io import BytesIO
from PIL import Image
import os
import torchvision.transforms as transforms  # Importar torchvision

app = Flask(__name__)

s3 = boto3.client('s3', 
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), 
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

BUCKET_NAME = 'modeloiasenas'
MODEL_KEY = 'abecedariobest.pt'

def load_model():
    model_file = BytesIO()
    s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_file)
    model_file.seek(0)
    model = torch.load(model_file, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    
    # Convertir la imagen a tensor
    transform = transforms.ToTensor()
    img_tensor = torch.unsqueeze(transform(img), 0)

    # Realiza la inferencia
    with torch.no_grad():
        results = model(img_tensor)

    # Suposición: El modelo devuelve un diccionario con etiquetas
    labels = [result['label'] for result in results]

    return jsonify({"labels": labels})

if __name__ == '__main__':
    app.run(debug=True)
