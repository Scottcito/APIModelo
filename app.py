from flask import Flask, request, jsonify
import boto3
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import os
import tempfile
import logging

app = Flask(__name__)

# Configurar logging para obtener más detalles de los errores
logging.basicConfig(level=logging.INFO)

# Configurar el cliente de S3 usando credenciales del entorno
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = 'modeloprueba'
MODEL_KEY = 'best.pt'

def load_model():
    try:
        logging.info("Intentando descargar el modelo desde S3.")
        model_file = BytesIO()
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_file)
        model_file.seek(0)
        logging.info("Modelo descargado con éxito.")

        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_model_file:
            temp_model_file.write(model_file.getbuffer())
            temp_model_path = temp_model_file.name
            logging.info(f"Modelo guardado temporalmente en {temp_model_path}.")

        # Cargar el modelo en el formato .pt
        model = YOLO(temp_model_path)
        logging.info("Modelo cargado con éxito.")
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}", exc_info=True)
        raise

# Cargar el modelo una vez al inicio
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file is None:
            logging.error("No se ha proporcionado ningún archivo.")
            return jsonify({"error": "No se ha proporcionado ningún archivo"}), 400

        img = Image.open(file.stream).convert('RGB')
        
        # Realizar la inferencia
        results = model(img)
        
        # Extraer solo los labels de los resultados
        labels = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                labels.append(label)
        
        logging.info(f"Predicción realizada con éxito. Labels: {labels}")
        return jsonify({"labels": labels})
    
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
