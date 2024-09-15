from flask import Flask, request, jsonify
import boto3, cv2, os, tempfile, logging
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Configuración S3
s3 = boto3.client('s3', 
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

BUCKET_NAME = 'modeloprueba'
MODELS = {'model_1': 'best.pt', 'model_2': 'best160Epocas.pt'}

def load_model(model_key):
    try:
        logging.info(f"Descargando modelo {model_key} de S3.")
        model_file = BytesIO()
        s3.download_fileobj(BUCKET_NAME, model_key, model_file)
        model_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
            temp_file.write(model_file.getbuffer())
        
        return YOLO(temp_file.name)
    except Exception as e:
        logging.error(f"Error cargando modelo {model_key}: {e}", exc_info=True)
        raise

model_1, model_2 = load_model(MODELS['model_1']), load_model(MODELS['model_2'])

def extract_labels(results, model, threshold=10):
    label_counts = {}
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            label_counts[label] = label_counts.get(label, 0) + 1
    return [label for label, count in label_counts.items() if count >= threshold]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = Image.open(request.files['file'].stream).convert('RGB')
        labels = extract_labels(model_1(img), model_1, threshold=1)
        return jsonify({"labels": labels})
    except Exception as e:
        logging.error(f"Error en /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    try:
        video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        request.files['file'].save(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("No se pudo abrir el archivo de video.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

        labels = extract_labels(model_2(frame), model_2)
        cap.release()
        return jsonify({"labels": labels})
    except Exception as e:
        logging.error(f"Error en /predict_video: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
