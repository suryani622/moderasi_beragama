from flask import Flask, request, jsonify
from flask_cors import CORS
from model import initialize_models, generate_response 
import json

app = Flask(__name__)
CORS(app)

try:
    print("Memulai inisialisasi model dan data...")
    initialize_models()
    print("Model siap.")
except Exception as e:
    print(f"[!] Gagal menginisialisasi model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({
            "success": False,
            "error": "Permintaan tidak valid."
        }), 400

    input_text = request.json['text']

    try:
        result = generate_response(input_text)

        return jsonify({
            "success": True,
            "input": input_text,
            "response": result["response_text"],
            "sentiment": result["detected_sentiment"]
        })
    
    except Exception as e:
        print(f"[!] Error saat prediksi: {e}")
        return jsonify({
            "success": False,
            "error": "Terjadi kesalahan internal saat memproses permintaan."
        }), 500

@app.route('/', methods=['GET'])
def home():
    return "Welcome!"

if __name__ == '__main__':
    print("Menjalankan Flask App di local server...")
    app.run(host='0.0.0.0', port=5000, debug=True)