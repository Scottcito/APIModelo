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
        file = request.files['file']
        if not file:
            logging.error("No se ha proporcionado ningún archivo.")
            return jsonify({"error": "No se ha proporcionado ningún archivo"}), 400

        # Guardar temporalmente el archivo de video
        video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        file.save(video_path)

        label_counts = {label: 0 for label in ["Buenos Días", "Hola", "Adiós", "Buenas Tardes", "Buenas Noches"]}
        label_thresholds = {
            "Buenos Días": 16,
            "Hola": 17,
            "Adiós": 10,
            "Buenas Tardes": 7,
            "Buenas Noches": 7,
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("No se pudo abrir el archivo de video.")
            os.remove(video_path)  # Eliminar el archivo de video temporal
            return jsonify({"error": "No se pudo abrir el archivo de video."}), 500

        frame_limit = 200  # Límite de frames
        frame_count = 0
        
        while cap.isOpened() and frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                logging.info("Se llegó al final del video o no se pudo leer el frame.")
                break
            frame_count += 1

            # Redimensionar el frame para acelerar la inferencia
            frame = cv2.resize(frame, (640, 480))

            # Realizar la inferencia con el segundo modelo
            results = model_2(frame)
            for result in results:
                for box in result.boxes:
                    label = model_2.names[int(box.cls)]
                    if label in label_counts:
                        label_counts[label] += 1
                        logging.info(f"Etiqueta '{label}' detectada, conteo actual: {label_counts[label]}")

                        # Verificar si alguna etiqueta alcanza el umbral
                        if label_counts[label] >= label_thresholds[label]:
                            cap.release()
                            os.remove(video_path)  # Eliminar el archivo de video temporal
                            logging.info(f"Se detectó la etiqueta '{label}' {label_thresholds[label]} veces. Deteniendo el procesamiento.")
                            return jsonify({"data": {"labels": [label]}})

        cap.release()
        os.remove(video_path)  # Eliminar el archivo de video temporal

        detected_labels = [label for label, count in label_counts.items() if count > 0]
        logging.info(f"Predicción con video realizada con éxito. Labels detectados: {detected_labels}")
        return jsonify({"data": {"labels": detected_labels}})

    except Exception as e:
        logging.error(f"Error durante la predicción con el video: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
