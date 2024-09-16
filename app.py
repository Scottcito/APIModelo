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
            "Buenas Tardes": 3,
            "Buenas Noches": 5,
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
