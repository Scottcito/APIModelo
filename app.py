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
MODEL_KEY_2 = 'best160Epocas.pt'

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
        img = Image.open(file.stream).convert('RGB')
        
        # Realizar la inferencia con el primer modelo
        results = model_1(img)
        
        # Extraer solo los labels
        labels = []
        for result in results:
            for box in result.boxes:
                label = model_1.names[int(box.cls)]
                labels.append(label)
        
        return jsonify({"labels": labels})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
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

        # Diccionario para contar las etiquetas
        label_counts = {}
        frequency_threshold = 3  # Umbral de frecuencia para filtrar etiquetas
        max_frames_to_process = 100  # Número máximo de frames a procesar

        # Leer el video frame por frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("No se pudo abrir el archivo de video.")
            return jsonify({"error": "No se pudo abrir el archivo de video."}), 500

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar la inferencia con el segundo modelo en cada frame
            results = model_2(frame)

            # Procesar los resultados y contar los labels detectados
            for result in results:
                for box in result.boxes:
                    label = model_2.names[int(box.cls)]
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1

            frame_count += 1
            if frame_count >= max_frames_to_process:
                logging.info(f"Se ha alcanzado el límite de {max_frames_to_process} frames procesados.")
                break

        cap.release()

        # Filtrar labels que superan el umbral de frecuencia
        filtered_labels = [label for label, count in label_counts.items() if count >= frequency_threshold]

        # Devolver los labels detectados
        logging.info(f"Predicción con video realizada con éxito. Labels: {filtered_labels}")
        return jsonify({"data": {"labels": filtered_labels}})

    except Exception as e:
        logging.error(f"Error durante la predicción con el video: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
