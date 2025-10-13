# File: model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Dot, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import re
import random

# ====================================================================
# BAGIAN 1: DEFINISI VARIABEL GLOBAL
# ====================================================================

# --- Variabel untuk Model Seq2Seq ---
SEQ2SEQ_WEIGHTS_PATH = "./bi_lstm.weights.h5"
SEQ2SEQ_TOKENIZER_PATH = 'tokenizer.pkl'
HIDDEN_DIM = 100
EMBED_DIM = 64
VOCAB_SIZE_SEQ2SEQ = 0 # Akan di-update saat tokenizer dimuat
maxlen_questions = 17
maxlen_answers = 25
tokenizer_seq2seq = None
index_to_word_seq2seq = None
START_TOKEN_WORD = 'start'
END_TOKEN_WORD = 'end'
encoder_model = None
decoder_model = None

# --- Variabel untuk Model Sentimen (BARU) ---
SENTIMENT_MODEL_PATH = 'model_sentimen.h5'
SENTIMENT_TOKENIZER_PATH = 'tokenizer_sentimen.pickle'
SENTIMENT_LABEL_ENCODER_PATH = 'label_encoder_sentimen.pickle'
MAX_LENGTH_SENTIMENT = 150
model_sentiment = None
tokenizer_sentiment = None
label_encoder = None

# ====================================================================
# BAGIAN 2: FUNGSI UNTUK MEMUAT ARTEFAK
# ====================================================================

def load_seq2seq_artifacts():
    """Memuat tokenizer untuk model Seq2Seq."""
    global VOCAB_SIZE_SEQ2SEQ, tokenizer_seq2seq, index_to_word_seq2seq
    if os.path.exists(SEQ2SEQ_TOKENIZER_PATH):
        try:
            with open(SEQ2SEQ_TOKENIZER_PATH, 'rb') as handle:
                tokenizer_seq2seq = pickle.load(handle)
            VOCAB_SIZE_SEQ2SEQ = len(tokenizer_seq2seq.word_index) + 1
            index_to_word_seq2seq = {index: word for word, index in tokenizer_seq2seq.word_index.items()}
            print(f"[*] Tokenizer Seq2Seq berhasil dimuat. VOCAB_SIZE: {VOCAB_SIZE_SEQ2SEQ}")
        except Exception as e:
            print(f"[!] ERROR: Gagal memuat tokenizer Seq2Seq: {e}")
    else:
        print(f"[!] WARNING: File tokenizer Seq2Seq tidak ditemukan di {SEQ2SEQ_TOKENIZER_PATH}")

def load_sentiment_artifacts():
    """Memuat model dan artefak untuk model Sentimen."""
    global model_sentiment, tokenizer_sentiment, label_encoder
    try:
        model_sentiment = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
        with open(SENTIMENT_TOKENIZER_PATH, 'rb') as handle:
            tokenizer_sentiment = pickle.load(handle)
        with open(SENTIMENT_LABEL_ENCODER_PATH, 'rb') as handle:
            label_encoder = pickle.load(handle)
        print("[*] Model Sentimen dan artefaknya berhasil dimuat.")
    except Exception as e:
        print(f"[!] ERROR: Gagal memuat artefak model sentimen: {e}")

# ====================================================================
# BAGIAN 3: FUNGSI-FUNGSI PREDIKSI
# ====================================================================

def preprocess_text_sentiment(text):
    """Preprocessing teks khusus untuk model sentimen."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    """Memprediksi sentimen dari sebuah kalimat."""
    if model_sentiment is None:
        return "netral" # Fallback jika model sentimen gagal dimuat
    
    cleaned_text = preprocess_text_sentiment(text)
    sequence = tokenizer_sentiment.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH_SENTIMENT, padding='post', truncating='post')
    prediction_probs = model_sentiment.predict(padded_sequence, verbose=0)
    predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_label

def str_to_tokens_seq2seq(sentence: str):
    """Mengubah kalimat string menjadi token untuk Seq2Seq."""
    words = sentence.lower().split()
    tokens_list = [tokenizer_seq2seq.word_index.get(w, 0) for w in words if tokenizer_seq2seq.word_index.get(w, 0) > 0]
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

def generate_seq2seq_response(input_text: str):
    """Fungsi PURE untuk menghasilkan respons dari model Seq2Seq."""
    tokens = str_to_tokens_seq2seq(input_text)
    enc_outs, h, c = encoder_model.predict(tokens, verbose=0)
    states_values = [h, c]
    empty_target_seq = np.zeros((1, 1))
    start_token_index = tokenizer_seq2seq.word_index.get(START_TOKEN_WORD, 1)
    empty_target_seq[0, 0] = start_token_index
    stop_condition = False
    decoded_sentence = ""
    max_decode_steps = maxlen_answers + 5
    step_count = 0
    while not stop_condition and step_count < max_decode_steps:
        dec_outputs, h, c = decoder_model.predict([empty_target_seq, enc_outs] + states_values, verbose=0)
        sampled_token_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = index_to_word_seq2seq.get(sampled_token_index)
        if sampled_word:
            if sampled_word == END_TOKEN_WORD or len(decoded_sentence.split()) >= maxlen_answers:
                stop_condition = True
            elif sampled_word != START_TOKEN_WORD:
                decoded_sentence += sampled_word + " "
        else:
            stop_condition = True
        empty_target_seq[0, 0] = sampled_token_index
        states_values = [h, c]
        step_count += 1
    return decoded_sentence.strip()

# ====================================================================
# BAGIAN 4: FUNGSI UTAMA YANG DIPANGGIL OLEH APP.PY (DIMODIFIKASI)
# ====================================================================

def generate_response(input_text: str):
    """
    Fungsi GATEKEEPER yang menggabungkan model sentimen dan seq2seq.
    Fungsi inilah yang akan dipanggil oleh app.py.
    """
    # 1. Prediksi sentimen terlebih dahulu
    sentiment = predict_sentiment(input_text)
    print(f"Pesan: '{input_text}', Sentimen Terdeteksi: {sentiment.upper()}")

    bot_response = ""

    # 2. Tentukan respons berdasarkan sentimen
    if sentiment == 'netral':
        # Jika netral, teruskan ke model seq2seq
        bot_response = generate_seq2seq_response(input_text)
        if not bot_response: # Fallback jika seq2seq tidak menghasilkan apa-apa
            bot_response = "Maaf, saya belum bisa menjawab pertanyaan itu. Coba tanya dengan cara lain ya."
    elif sentiment == 'negatif':
        # Jika negatif, berikan respons de-eskalasi
        responses = [
            "Mohon maaf jika ada kekurangan dari saya. Mari kita tetap gunakan bahasa yang baik. Anda akan dikenakan pelanggaran jika terus berkata tidak baik",
            "Saya mengerti Anda mungkin merasa frustrasi. Ada yang bisa saya bantu jelaskan kembali?",
            "Saya di sini untuk membantu. Mohon sampaikan masukan Anda dengan baik agar bisa saya proses. Anda akan dikenakan pelanggaran jika terus berkata tidak baik"
        ]
        bot_response = random.choice(responses)
    elif sentiment == 'positif':
        # Jika positif, berikan respons ramah
        responses = [
            "Terima kasih atas apresiasinya! Senang bisa membantu.",
            "Terima kasih banyak! Ada lagi yang bisa saya bantu?",
            "Senang mendengarnya! Semoga informasi ini bermanfaat."
        ]
        bot_response = random.choice(responses)

    return {
        "response_text": bot_response,
        "detected_sentiment": sentiment
    }

# ====================================================================
# BAGIAN 5: INISIALISASI SEMUA MODEL
# ====================================================================

# (Bagian ini tidak berubah, hanya isinya yang sedikit disesuaikan)
def create_model_architecture(vocab_size, embed_dim=EMBED_DIM, training=False):
    enc_inputs = Input(shape=(None,), name='encoder_input')
    enc_embedding_layer = Embedding(vocab_size, embed_dim, mask_zero=True, name='enc_embedding')
    enc_embedding = enc_embedding_layer(enc_inputs)
    enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(
        LSTM(HIDDEN_DIM, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0),
        name='encoder_bilstm'
    )(enc_embedding)
    state_h = Concatenate(name='concat_h')([forward_h, backward_h])
    state_c = Concatenate(name='concat_c')([forward_c, backward_c])
    enc_states = [state_h, state_c]
    dec_inputs = Input(shape=(None,), name='decoder_input')
    dec_embedding_layer = Embedding(vocab_size, embed_dim, mask_zero=True, name='dec_embedding')
    dec_embedding = dec_embedding_layer(dec_inputs)
    dec_lstm = LSTM(HIDDEN_DIM * 2, return_sequences=True, return_state=True, dropout=0.0, recurrent_dropout=0.0, name='decoder_lstm')
    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)
    attention = Dot(axes=[2, 2], name='attention_score')([dec_outputs, enc_outputs])
    attention = Activation('softmax', name='attention_weights')(attention)
    context = Dot(axes=[2, 1], name='context_vector')([attention, enc_outputs])
    dec_combined_context = Concatenate(axis=-1, name='combined_context')([context, dec_outputs])
    dec_dense = Dense(vocab_size, activation='softmax', name='output_layer')
    output = dec_dense(dec_combined_context)
    full_model = Model(inputs=[enc_inputs, dec_inputs], outputs=output)
    return full_model, enc_inputs, enc_outputs, enc_states, dec_inputs, dec_embedding_layer, dec_lstm, dec_dense

def make_inference_models():
    global VOCAB_SIZE_SEQ2SEQ
    full_model, enc_inputs, enc_outputs, enc_states, dec_inputs, dec_embedding_layer, dec_lstm, dec_dense = create_model_architecture(VOCAB_SIZE_SEQ2SEQ)
    if os.path.exists(SEQ2SEQ_WEIGHTS_PATH):
        try:
            full_model.load_weights(SEQ2SEQ_WEIGHTS_PATH)
            print(f"[*] Weights Seq2Seq berhasil dimuat dari {SEQ2SEQ_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[!] ERROR: Gagal memuat weights model Seq2Seq: {e}")
    enc_model = Model(enc_inputs, [enc_outputs] + enc_states)
    enc_out_input = Input(shape=(None, HIDDEN_DIM*2))
    dec_state_input_h = Input(shape=(HIDDEN_DIM*2,))
    dec_state_input_c = Input(shape=(HIDDEN_DIM*2,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_emb2 = dec_embedding_layer(dec_inputs)
    dec_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2, initial_state=dec_states_inputs)
    attention2 = Dot(axes=[2, 2])([dec_outputs2, enc_out_input])
    attention2 = Activation('softmax')(attention2)
    context2 = Dot(axes=[2, 1])([attention2, enc_out_input])
    dec_combined_context2 = Concatenate(axis=-1)([context2, dec_outputs2])
    dec_outputs2 = dec_dense(dec_combined_context2)
    dec_model = Model([dec_inputs, enc_out_input] + dec_states_inputs, [dec_outputs2, state_h2, state_c2])
    return enc_model, dec_model

def initialize_models():
    """Menginisialisasi SEMUA model dan artefak yang dibutuhkan."""
    global encoder_model, decoder_model

    print("--- 1. Memuat Artefak Model Sentimen ---")
    load_sentiment_artifacts()

    print("--- 2. Memuat Artefak Model Seq2Seq ---")
    load_seq2seq_artifacts()

    print("--- 3. Membangun dan Memuat Model Seq2Seq Keras ---")
    if VOCAB_SIZE_SEQ2SEQ > 0:
        encoder_model, decoder_model = make_inference_models()
        print("[*] Inisialisasi Semua Model Selesai.")
    else:
        print("[!] ERROR: VOCAB_SIZE Seq2Seq belum terdefinisi. Model Seq2Seq gagal dimuat.")