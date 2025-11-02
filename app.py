import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf
import io
from st_audiorec import st_audiorec

TARGET_SR = 22050
SILENCE_THRESHOLD = 30

def preprocess_audio(y, sr):
    """
    Memuat data audio (dari memory), trim, dan normalisasi.
    """
    try:
        # 1. Trimming Keheningan
        y_trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_THRESHOLD)
        
        # 2. Normalisasi
        if np.max(np.abs(y_trimmed)) > 0:
            y_normalized = librosa.util.normalize(y_trimmed)
        else:
            y_normalized = y_trimmed
            
        return y_normalized, sr
        
    except Exception as e:
        st.error(f"Error memproses audio: {e}")
        return None, None

def extract_features(y, sr):
    """
    Mengekstrak 4 fitur statistik yang diminta dan mengagregasinya.
    """
    features = {}
    
    # 1. Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # 2. Root Mean Square Energy (RMS)
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_cent_mean'] = np.mean(spectral_centroid)
    features['spec_cent_std'] = np.std(spectral_centroid)
    
    # 4. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spec_bw_mean'] = np.mean(spectral_bandwidth)
    features['spec_bw_std'] = np.std(spectral_bandwidth)
    
    # Mengembalikan dalam bentuk list sesuai urutan fitur saat pelatihan
    ordered_features = [
        features['zcr_mean'], features['zcr_std'],
        features['rms_mean'], features['rms_std'],
        features['spec_cent_mean'], features['spec_cent_std'],
        features['spec_bw_mean'], features['spec_bw_std']
    ]
    
    return ordered_features

# --- Fungsi Pemuatan Model ---

@st.cache_resource
def load_model(model_path):
    """
    Memuat model dan scaler. 
    @st.cache_resource memastikan ini hanya berjalan sekali.
    """
    try:
        data = joblib.load(model_path)
        return data['model'], data['scaler'], data['features']
    except FileNotFoundError:
        st.error(f"File model '{model_path}' tidak ditemukan.")
        st.error("Pastikan Anda sudah menjalankan '2_train_model.py'.")
        return None, None, None

# --- Antarmuka Aplikasi Streamlit ---

# Muat model
model, scaler, feature_names = load_model('audio_classifier.pkl')

st.set_page_config(layout="wide")
st.title("🎙️ Classifier Perintah Suara")
st.markdown("### (Buka / Tutup)")
st.write("Tekan tombol rekam, ucapkan **'buka'** atau **'tutup'**, lalu tekan tombol stop.")

if model is not None:
    
    # 1. Tampilkan Perekam Audio
    wav_audio_data = st_audiorec()

    # Inisialisasi placeholder untuk hasil
    result_placeholder = st.empty()

    if wav_audio_data is not None:
        result_placeholder.info("Menganalisis audio...")
        
        try:
            # 2. Memuat audio dari bytes yang direkam
            # st_audiorec mengembalikan data audio dalam format WAV
            audio_bytes = io.BytesIO(wav_audio_data)
            y, sr = librosa.load(audio_bytes, sr=TARGET_SR)
            
            # 3. Pra-pemrosesan
            y_proc, sr_proc = preprocess_audio(y, sr)
            
            if y_proc is not None and len(y_proc) > 0:
                # 4. Ekstraksi Fitur
                features = extract_features(y_proc, sr_proc)
                
                # 5. Penskalaan Fitur
                # Ubah ke 2D array (karena scaler mengharapkan batch)
                features_scaled = scaler.transform([features])
                
                # 6. Prediksi
                prediction = model.predict(features_scaled)
                prediction_proba = model.predict_proba(features_scaled)
                
                # Dapatkan probabilitas untuk kelas yang diprediksi
                confidence = np.max(prediction_proba) * 100
                
                # 7. Tampilkan Hasil
                command = prediction[0].upper()
                
                if command == "BUKA":
                    result_placeholder.success(f"**Prediksi: {command}** (Confidence: {confidence:.2f}%)")
                elif command == "TUTUP":
                    result_placeholder.error(f"**Prediksi: {command}** (Confidence: {confidence:.2f}%)")
                else:
                    result_placeholder.warning(f"**Prediksi: {command}** (Confidence: {confidence:.2f}%)")
                
                # Tampilkan audio yang diputar ulang (opsional)
                st.audio(wav_audio_data, format='audio/wav')

            else:
                result_placeholder.warning("Tidak ada audio yang terdeteksi setelah trimming keheningan. Coba bicara lebih keras.")
        
        except Exception as e:
            result_placeholder.error(f"Terjadi kesalahan saat pemrosesan: {e}")

else:
    st.error("Aplikasi tidak dapat dimuat karena model tidak ditemukan.")