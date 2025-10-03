from flask import Flask, request, jsonify
from flask_cors import CORS
# Mengimpor fungsi yang benar: initialize_models dan generate_response
from model import initialize_models, generate_response 
import json

# ====================================================================
# 1. INISIALISASI FLASK
# ====================================================================

app = Flask(__name__)
CORS(app)

# ====================================================================
# 2. ENDPOINT PREDIKSI
# ====================================================================

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

# ====================================================================
# 3. UTILITY DAN RUN
# ====================================================================

@app.route('/', methods=['GET'])
def home():
    """Endpoint home sederhana."""
    return "API Seq2Seq Attention aktif. Gunakan endpoint /predict (POST) dengan payload {'text': '...'}"

if __name__ == '__main__':
    # Inisialisasi model hanya SEKALI saat server dimulai
    print("Memulai inisialisasi model dan data...")
    initialize_models()
    print("Model siap. Menjalankan Flask App...")
    
    app.run(host='0.0.0.0', port=5000)