from flask import Flask, request, jsonify
import boto3
import cv2
import os
import tempfile
import logging
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configurar cliente S3
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Configuración de los modelos
BUCKET_NAME = 'modeloprueba'
MODEL_KEY_1 = 'best.pt'
MODEL_KEY_2 = 'palabrasbest.pt'

def load_model(model_key):
    try:
        logging.info(f"Intentando descargar el modelo {model_key} desde S3.")
        model_file = BytesIO()
        s3.download_fileobj(BUCKET_NAME, model_key, model_file)
        model_file.seek(0)
        logging.info("Modelo descargado con éxito.")

        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_model_file:
            temp_model_file.write(model_file.getbuffer())
            temp_model_path = temp_model_file.name
            logging.info(f"Modelo guardado temporalmente en {temp_model_path}.")

        model = YOLO(temp_model_path)
        logging.info(f"Modelo {model_key} cargado con éxito.")
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo {model_key}: {e}", exc_info=True)
        raise

# Cargar los modelos
model_1 = load_model(MODEL_KEY_1)
model_2 = load_model(MODEL_KEY_2)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file is None:
            logging.error("No se ha proporcionado ningún archivo.")
            return jsonify({"error": "No se ha proporcionado ningún archivo"}), 400

        img = Image.open(file.stream).convert('RGB')
        
        # Realizar la inferencia con el primer modelo
        results = model_1(img)
        
        # Extraer solo los labels de los resultados
        labels = []
        for result in results:
            for box in result.boxes:
                label = model_1.names[int(box.cls)]
                labels.append(label)
        
        logging.info(f"Predicción realizada con éxito. Labels: {labels}")
        return jsonify({"labels": labels})
    
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    try:
        # Obtener el archivo de video
        file = request.files['file']
        if file is None:
            logging.error("No se ha proporcionado ningún archivo.")
            return jsonify({"error": "No se ha proporcionado ningún archivo"}), 400

        # Guardar temporalmente el archivo de video
        video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        file.save(video_path)

        labels = []

        # Leer el video frame por frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("No se pudo abrir el archivo de video.")
            return jsonify({"error": "No se pudo abrir el archivo de video."}), 500

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar la inferencia con el segundo modelo en cada frame
            results = model_2(frame)

            # Procesar los resultados y agregar los labels detectados
            if results:
                for result in results:
                    for box in result.boxes:
                        label = model_2.names[int(box.cls)]
                        if label not in labels:
                            labels.append(label)

                        # Si ya se encontró un label, detener el procesamiento
                        if labels:
                            cap.release()  # Cerrar el video
                            logging.info(f"Predicción realizada con éxito. Labels: {labels}")
                            return jsonify({"labels": labels})

        cap.release()

        # Devolver los labels detectados (si no se encontraron, devuelve una lista vacía)
        logging.info(f"Predicción con video realizada con éxito. Labels: {labels}")
        return jsonify({"labels": labels})

    except Exception as e:
        logging.error(f"Error durante la predicción con el video: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
