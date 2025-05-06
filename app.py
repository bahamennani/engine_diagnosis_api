import traceback

from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import cv2
import uuid
import os
import tensorflow as tf
import pickle
import gc
from pydub import AudioSegment

app = Flask(__name__)

# إعداد المسارات
MODELS = {
    "essence": {
        "model": "models/cnn_model_essence.tflite",
        "labels": ["Healthy engine", "Raté d'allumage", "Consommation d'huile", "Ralenti instable"]
    },
    "diesel": {
        "model": "models/cnn_model_diesel.tflite",
        "labels": ["Healthy engine", "Damaged engine"]
    }
}

IMG_SIZE = (128, 128)

# تحويل الصوت إلى spectrogram
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.56, 2.56), dpi=50)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='gray_r')

    tmp_image_path = f"{uuid.uuid4()}.png"
    plt.savefig(tmp_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return tmp_image_path

# معالجة الصورة
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype(np.float32)
    return img

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files or "engine_type" not in request.form:
        return jsonify({"error": "Audio file and engine type are required"}), 400

    engine_type = request.form["engine_type"]
    if engine_type not in MODELS:
        return jsonify({"error": "Invalid engine type"}), 400

    audio_file = request.files["audio"]
    filename = audio_file.filename.lower()

    original_path = f"{uuid.uuid4()}"
    image_path = None

    try:
        # التحقق من نوع الملف (AAC أو WAV)
        if filename.endswith(".aac"):
            aac_path = f"{original_path}.aac"
            wav_path = f"{original_path}.wav"
            audio_file.save(aac_path)

            # تحويل AAC إلى WAV
            audio = AudioSegment.from_file(aac_path, format="aac")
            audio.export(wav_path, format="wav")
        elif filename.endswith(".wav"):
            wav_path = f"{original_path}.wav"
            audio_file.save(wav_path)
        else:
            return jsonify({"error": "Unsupported audio format. Please upload .wav or .aac file"}), 400

        # تحويل WAV إلى spectrogram
        image_path = audio_to_spectrogram(wav_path)
        img = preprocess_image(image_path)

        # تحميل النموذج
        interpreter = tf.lite.Interpreter(model_path=MODELS[engine_type]["model"])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = np.argmax(output_data)
        prediction = MODELS[engine_type]["labels"][predicted_class_index]
        confidence = float(output_data[0][predicted_class_index])

        return jsonify({"prediction": prediction, "confidence": round(confidence, 2)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

    finally:
        # حذف الملفات المؤقتة
        for ext in [".aac", ".wav", ".png"]:
            file_path = f"{original_path}{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)
        plt.close('all')
        gc.collect()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
