import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from flask import Flask, request, render_template, jsonify

# Konsoldaki gereksiz TensorFlow uyarılarını gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_YOLU = os.environ.get("MODEL_YOLU", "bebek_uyku_modeli.h5")
SINIF_YOLU = os.environ.get("SINIF_YOLU", "classes.npy")
YAMNET_URL = os.environ.get("YAMNET_URL", "https://tfhub.dev/google/yamnet/1")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB

model = None
classes = None
yamnet_model = None


def modeli_yukle():
    """Model, sınıflar ve YAMNet'i tembel (lazy) olarak yükler."""
    global model, classes, yamnet_model

    if not os.path.exists(MODEL_YOLU):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_YOLU}. "
            f"Lütfen 'bebek_uyku_modeli.h5' dosyasını proje köküne (veya MODEL_YOLU ile "
            f"belirtilen yola) yerleştirin."
        )
    if not os.path.exists(SINIF_YOLU):
        raise FileNotFoundError(
            f"Sınıf dosyası bulunamadı: {SINIF_YOLU}. "
            f"Lütfen 'classes.npy' dosyasını proje köküne (veya SINIF_YOLU ile "
            f"belirtilen yola) yerleştirin."
        )

    if model is None:
        print("YAMNet yükleniyor...")
        yamnet_model = hub.load(YAMNET_URL)

        print("Model yükleniyor...")
        model = tf.keras.models.load_model(
            MODEL_YOLU, custom_objects={"KerasLayer": hub.KerasLayer}
        )
        classes = np.load(SINIF_YOLU, allow_pickle=True)
        print("Model ve YAMNet hazır.")


def load_and_resample_audio(file_path):
    audio_numpy, _ = librosa.load(file_path, sr=16000, mono=True)
    return tf.convert_to_tensor(audio_numpy, dtype=tf.float32)


def ses_analiz_et(file_path):
    modeli_yukle()

    wav_data = load_and_resample_audio(file_path)
    _, embeddings, _ = yamnet_model(wav_data)

    if embeddings.shape[0] == 0:
        raise ValueError("Ses dosyasından anlamlı özellikler çıkarılamadı.")

    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    mean_embedding = np.expand_dims(mean_embedding, axis=0)

    cry_pred, reason_pred = model.predict(mean_embedding, verbose=0)

    crying_probability = float(cry_pred[0][0])
    is_crying = crying_probability > 0.5

    probabilities = []
    for idx, class_name in enumerate(classes):
        probabilities.append(
            {"label": str(class_name), "probability": float(reason_pred[0][idx])}
        )
    probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "is_crying": bool(is_crying),
        "crying_probability": crying_probability,
        "top_reason": probabilities[0]["label"] if probabilities else None,
        "probabilities": probabilities,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı. 'file' alanı zorunludur."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Dosya seçilmedi."}), 400

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = ses_analiz_et(tmp_path)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Analiz sırasında hata oluştu: {e}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
