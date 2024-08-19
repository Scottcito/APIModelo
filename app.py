from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Permite solicitudes de otros orígenes

model = YOLO("models/abecedariobest.pt")  

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file selected'}), 400

    image = Image.open(BytesIO(file.read()))

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(image, imgsz=640, conf=0.5)  # Ajusta la confianza aquí

    labels = [model.names[int(box[-1])] for box in results[0].boxes.data]

    return jsonify({'labels': labels})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Asegúrate de que Flask esté escuchando en todas las interfaces
