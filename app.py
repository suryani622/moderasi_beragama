from flask import Flask, request, jsonify
from flask_cors import CORS
from model import initialize_models, generate_response 
import json

app = Flask(__name__)
CORS(app)

# Inisialisasi model hanya SEKALI saat aplikasi dimuat.
# Ini memastikan Gunicorn akan memuat model saat startup.
try:
    print("Memulai inisialisasi model dan data...")
    initialize_models()
    print("Model siap.")
except Exception as e:
    print(f"[!] Gagal menginisialisasi model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk menerima teks input dan mengembalikan respons model.
    Input JSON: {"text": "Bagaimana kabarmu?"}
    """
    if not request.json or 'text' not in request.json:
        return jsonify({
            "success": False,
            "error": "Permintaan tidak valid. Harap kirimkan JSON dengan kunci 'text'."
        }), 400

    input_text = request.json['text']

    try:
        # Panggil fungsi prediksi
        response_text = generate_response(input_text)

        return jsonify({
            "success": True,
            "input": input_text,
            "response": response_text
        })
    
    except Exception as e:
        print(f"[!] Error saat prediksi: {e}")
        return jsonify({
            "success": False,
            "error": "Terjadi kesalahan internal saat memproses permintaan."
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Endpoint home sederhana."""
    return "Welcome!"

if __name__ == '__main__':
    # Blok ini hanya untuk menjalankan server pengembangan (development)
    # Gunicorn akan mengabaikan ini
    print("Menjalankan Flask App di local server...")
    app.run(host='0.0.0.0', port=5000, debug=True)