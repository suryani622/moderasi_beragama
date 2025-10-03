import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Dot, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np
import os
import pickle

# ====================================================================
# KONSTANTA DAN PARAMETER MODEL
# ====================================================================

# Parameter yang sama seperti yang Anda definisikan
HIDDEN_DIM = 100
EMBED_DIM = 64
DROPOUT = 0.2
WEIGHTS_PATH = "./bi_lstm.weights.h5"
TOKENIZER_PATH = 'tokenizer.pkl' # Asumsi Anda menyimpan tokenizer

# Variabel Global untuk Inference Model
encoder_model = None
decoder_model = None

# ====================================================================
# 1. PARAMETER DATA (WAJIB DIISI SEBELUM DEPLOYMENT)
# *** SINKRONISASI TOKEN DENGAN PREPROCESSING COLAB <START> dan <END> ***
# ====================================================================
VOCAB_SIZE = 1133        # Ganti dengan perkiraan awal atau nilai final Vokabulari
maxlen_questions = 17     # <-- GANTI DENGAN NILAI MAXLEN QUESTIONS YANG TEPAT
maxlen_answers = 25    # <-- GANTI DENGAN NILAI MAXLEN ANSWERS YANG TEPAT
tokenizer = None          # Objek Keras Tokenizer Anda
index_to_word = None      # Dictionary {index: word}
START_TOKEN_WORD = 'start' # Word yang Benar Sesuai Preprocessing
END_TOKEN_WORD = 'end'     # Word yang Benar Sesuai Preprocessing

def load_data_artifacts():
    """Memuat tokenizer dan data preprocessing lainnya."""
    global VOCAB_SIZE, maxlen_questions, maxlen_answers, tokenizer, index_to_word

    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
                VOCAB_SIZE = len(tokenizer.word_index) + 1
                print(f"[*] Tokenizer berhasil dimuat. VOCAB_SIZE final: {VOCAB_SIZE}")
                print("Tokenizer sample word_index:", list(tokenizer.word_index.items())[:10])
                index_to_word = {index: word for word, index in tokenizer.word_index.items()}
                index_to_word = {index: word for word, index in tokenizer.word_index.items()}
                print("Sample index_to_word keys:", list(index_to_word.keys())[:10])
                print("Sample index_to_word values:", list(index_to_word.values())[:10])

        except Exception as e:
            print(f"ERROR: Gagal memuat tokenizer: {e}. Menggunakan placeholder default.")
            # Fallback jika loading gagal
            VOCAB_SIZE = 1133
            # Fallback menggunakan indeks yang sudah dikonfirmasi: 1=<START>, 2=<END>
            tokenizer_mock = { 'word_index': { START_TOKEN_WORD: 1, END_TOKEN_WORD: 2, 'hai': 3 } }
            index_to_word = {v: k for k, v in tokenizer_mock['word_index'].items()}
            tokenizer = type('Tokenizer', (object,), {'word_index': tokenizer_mock['word_index']})()
            

# ====================================================================
# 2. DEFINISI ARSITEKTUR MODEL
# ====================================================================

def create_model_architecture(vocab_size, embed_dim=EMBED_DIM, dropout_rate=DROPOUT, training=True):
    """
    Membangun arsitektur model Seq2Seq Attention.
    """
    # Encoder
    enc_inputs = Input(shape=(None,), name='encoder_input')
    enc_embedding_layer = Embedding(vocab_size, embed_dim, mask_zero=True, name='enc_embedding')
    enc_embedding = enc_embedding_layer(enc_inputs)
    enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(
        LSTM(HIDDEN_DIM, return_sequences=True, return_state=True, dropout=dropout_rate if training else 0.0, recurrent_dropout=dropout_rate if training else 0.0),
        name='encoder_bilstm'
    )(enc_embedding)

    state_h = Concatenate(name='concat_h')([forward_h, backward_h])
    state_c = Concatenate(name='concat_c')([forward_c, backward_c])
    enc_states = [state_h, state_c]

    # Decoder
    dec_inputs = Input(shape=(None,), name='decoder_input')
    dec_embedding_layer = Embedding(vocab_size, embed_dim, mask_zero=True, name='dec_embedding')
    dec_embedding = dec_embedding_layer(dec_inputs)
    
    dec_lstm = LSTM(HIDDEN_DIM * 2, return_sequences=True, return_state=True, dropout=dropout_rate if training else 0.0, recurrent_dropout=dropout_rate if training else 0.0, name='decoder_lstm')
    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

    # Attention
    attention = Dot(axes=[2, 2], name='attention_score')([dec_outputs, enc_outputs])
    attention = Activation('softmax', name='attention_weights')(attention)
    context = Dot(axes=[2, 1], name='context_vector')([attention, enc_outputs])

    # Merge context + decoder output
    dec_combined_context = Concatenate(axis=-1, name='combined_context')([context, dec_outputs])
    dec_combined_context = Dropout(dropout_rate if training else 0.0)(dec_combined_context)

    # Final prediction
    dec_dense = Dense(vocab_size, activation='softmax', name='output_layer')
    output = dec_dense(dec_combined_context)

    full_model = Model(inputs=[enc_inputs, dec_inputs], outputs=output)
    
    return full_model, enc_inputs, enc_outputs, enc_states, dec_inputs, dec_embedding_layer, dec_lstm, dec_dense

# ====================================================================
# 3. FUNGSI INFERENCE MODEL
# ====================================================================

def make_inference_models():
    """Membuat model Encoder dan Decoder terpisah untuk inference tanpa dropout."""
    
    # 1. Bangun arsitektur model lengkap tanpa dropout (training=False)
    full_model, enc_inputs, enc_outputs, enc_states, dec_inputs, dec_embedding_layer, dec_lstm, dec_dense = create_model_architecture(VOCAB_SIZE, training=False)
    
    # 2. Load weights
    if os.path.exists(WEIGHTS_PATH):
        try:
            full_model.load_weights(WEIGHTS_PATH)
            print(f"[*] Weights berhasil dimuat dari {WEIGHTS_PATH}")
        except Exception as e:
             print(f"ERROR: Gagal memuat weights model: {e}")

    # --- ENCODER MODEL ---
    enc_model = Model(enc_inputs, [enc_outputs] + enc_states)

    # --- DECODER MODEL ---
    enc_out_input = Input(shape=(None, HIDDEN_DIM*2), name='encoder_output_inf')
    dec_state_input_h = Input(shape=(HIDDEN_DIM*2,), name='decoder_state_h_inf')
    dec_state_input_c = Input(shape=(HIDDEN_DIM*2,), name='decoder_state_c_inf')
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]

    # Menggunakan LAYER Embedding yang sudah memuat weights
    dec_emb2 = dec_embedding_layer(dec_inputs) 
    
    # LSTM Decoder tanpa dropout (sudah build tanpa dropout)
    dec_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2, initial_state=dec_states_inputs)

    # Attention di dalam decoder inference
    attention2 = Dot(axes=[2, 2])([dec_outputs2, enc_out_input])
    attention2 = Activation('softmax')(attention2)
    context2 = Dot(axes=[2, 1])([attention2, enc_out_input])
    dec_combined_context2 = Concatenate(axis=-1)([context2, dec_outputs2])

    # Dense Layer (menggunakan objek layer yang sama)
    dec_outputs2 = dec_dense(dec_combined_context2)

    dec_model = Model(
        [dec_inputs, enc_out_input] + dec_states_inputs,
        [dec_outputs2, state_h2, state_c2]
    )

    return enc_model, dec_model 

# ====================================================================
# 4. PREDIKSI UTAMA UNTUK API
# ====================================================================

def str_to_tokens(sentence: str):
    """Mengubah kalimat string menjadi sekuens token dan padding."""
    if tokenizer is None or tokenizer.word_index is None:
        raise ValueError("Tokenizer belum dimuat. Panggil load_data_artifacts() terlebih dahulu.")
        
    words = sentence.lower().split()
    tokens_list = [tokenizer.word_index.get(w, 0) for w in words if tokenizer.word_index.get(w, 0) > 0]
    return pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')

def generate_response(input_text: str):
    """Fungsi utama untuk menghasilkan respons dari teks input."""
    global encoder_model, decoder_model, index_to_word
    
    if encoder_model is None or decoder_model is None:
        return "Model belum diinisialisasi. Panggil initialize_models() terlebih dahulu."

    # 1. Encoder step: Dapatkan state awal dari input
    try:
        tokens = str_to_tokens(input_text)
        print("Input tokens:", tokens)  # Debug: cek token input
        enc_outs, h, c = encoder_model.predict(tokens, verbose=0)
        states_values = [h, c]
    except Exception as e:
        print(f"Error during encoder prediction: {e}")
        return "Terjadi kesalahan saat memproses input oleh encoder."

    # 2. Inisialisasi Decoder: Mulai dengan token '<START>'
    empty_target_seq = np.zeros((1, 1))
    
    # MENGGUNAKAN INDEKS YANG SUDAH DISINKRONKAN: <START>
    start_token_index = tokenizer.word_index.get(START_TOKEN_WORD) 
    end_token_word = END_TOKEN_WORD
    
    # Jika index START tidak ditemukan (misalnya, tokenizer hanya berisi data mock), gunakan 1
    if start_token_index is None:
        start_token_index = 1
        
    empty_target_seq[0, 0] = start_token_index 

    stop_condition = False
    decoded_sentence = ""
    
    # Batas keamanan untuk menghindari loop tak terbatas
    max_decode_steps = maxlen_answers + 5 
    step_count = 0

    # 3. Loop Decoding
    while not stop_condition and step_count < max_decode_steps:
        # Decoder step
        dec_outputs, h, c = decoder_model.predict([empty_target_seq, enc_outs] + states_values, verbose=0)
        sampled_token_index = np.argmax(dec_outputs[0, -1, :])
        
        # Dapatkan kata dari index
        sampled_word = index_to_word.get(sampled_token_index)
        print(f"Step {step_count}: sampled_token_index={sampled_token_index}, sampled_word={sampled_word}")


        if sampled_word is not None:
            # Pengecekan KONDISI BERHENTI (membandingkan dengan <END>)
            if sampled_word == end_token_word or len(decoded_sentence.split()) >= maxlen_answers:
                stop_condition = True
            elif sampled_word != START_TOKEN_WORD: 
                decoded_sentence += sampled_word + " "
        else:
            # Jika memprediksi token OOV/Padding (Index 0), hentikan loop
            stop_condition = True

        # Update input sequence dan states
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_token_index
        states_values = [h, c]
        step_count += 1
        
    return decoded_sentence.strip()

# ====================================================================
# 5. INISIALISASI MODEL (Hanya sekali saat startup)
# ====================================================================

def initialize_models():
    """Menginisialisasi dan memuat semua artefak yang dibutuhkan."""
    global encoder_model, decoder_model

    print("--- Memuat Artefak Data ---")
    load_data_artifacts()

    print("--- Memuat Model Keras ---")
    if VOCAB_SIZE is not None and VOCAB_SIZE > 0:
        encoder_model, decoder_model = make_inference_models()
        print("[*] Inisialisasi Model Selesai.")
    else:
        print("[!] ERROR: VOCAB_SIZE belum terdefinisi. Model gagal dimuat.")
