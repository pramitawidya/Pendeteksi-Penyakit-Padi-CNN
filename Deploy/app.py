import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import io

# Membuat instance Flask
app = Flask(__name__)
CORS(app)

# Memuat model
model = tf.keras.models.load_model('riceleaf_desease_model.h5')  # Ganti dengan nama model Anda

class_indices = {
    0: 'Hawar daun bakteri',
    1: 'Bercak Coklat',
    2: 'Sehat',
}

# Fungsi untuk memproses gambar langsung dari file
def process_image(file, target_size=(224, 224)):
    try:
        # Buka file gambar menggunakan PIL
        image = Image.open(io.BytesIO(file))
        # Ubah ukuran gambar
        image = image.resize(target_size)
        # Konversi gambar menjadi array dan normalisasi
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/')
def index():
    return "Hello World"  # API root yang mengembalikan pesan "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    # Mengecek apakah file diupload
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Proses gambar langsung dari file
        processed_image = process_image(file.read())  # Baca file sebagai stream byte

        # Melakukan prediksi
        predictions = model.predict(processed_image)
        class_idx = np.argmax(predictions, axis=1)[0]  # Ambil indeks kelas dengan probabilitas tertinggi
        confidence = np.max(predictions)  # Probabilitas tertinggi

        class_name = class_indices.get(class_idx, 'Unknow')

        # Menyusun hasil prediksi sebagai JSON
        result = {
            'class_name': class_name,
            'class_idx': int(class_idx),
            'confidence': float(confidence)
        }

        return jsonify(result)  # Kembalikan hasil prediksi dalam format JSON
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    # Menjalankan aplikasi Flask
    app.run(debug=True)