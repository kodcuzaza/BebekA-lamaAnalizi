import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Konsoldaki gereksiz TensorFlow uyarılarını gizleyelim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("1. Modeli ve Etiket Düzenini Yükleme...")
# Eğittiğimiz modeli ve sınıfları çağırıyoruz
model = tf.keras.models.load_model("bebek_uyku_modeli.h5")
classes = np.load("classes.npy", allow_pickle=True)

print("2. YAMNet Özellik Çıkarıcı Yükleniyor...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_and_resample_audio(file_path):
    # Sesi YAMNet'in istediği formata getirir
    audio_numpy, sr = librosa.load(file_path, sr=16000, mono=True)
    audio = tf.convert_to_tensor(audio_numpy, dtype=tf.float32)
    return audio

def ses_analiz_et(file_path):
    if not os.path.exists(file_path):
        print(f"\nHata: '{file_path}' yolunda bir dosya bulunamadı! Lütfen yolu kontrol edin.")
        return
    
    print("\nSes yükleniyor ve yapay zekaya gönderiliyor...")
    wav_data = load_and_resample_audio(file_path)
    
    # YAMNet imzasını alalım
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    
    if embeddings.shape[0] == 0:
        print("Hata: Ses dosyasından anlamlı özellikler çıkarılamadı.")
        return
        
    # Sesin ortalama embedding değerini hesapla ve boyutu ayarla
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    mean_embedding = np.expand_dims(mean_embedding, axis=0) # (1, 1024) yapıyoruz
    
    # Modelimizden tahmini alıyoruz (Çift çıktı verecek: Ağlama ve Neden)
    cry_pred, reason_pred = model.predict(mean_embedding, verbose=0)
    
    # 1. ÇIKTI: Ağlama durumu kontrolü
    crying_probability = cry_pred[0][0]
    is_crying = crying_probability > 0.5
    
    print("\n=======================================================")
    print("                     ANALİZ SONUCU                     ")
    print("=======================================================")
    if is_crying:
        print(f"DURUM: 😭 Bebek Ağlıyor! (Eminlik: %{crying_probability*100:.2f})")
    else:
        print(f"DURUM: 😊 Bebek Ağlamıyor. (Eminlik: %{(1 - crying_probability)*100:.2f})")
    
    # 2. ÇIKTI: Detaylı ihtimaller (Açlık, yorgunluk, gülme vb.)
    print("\nTÜM OLASILIKLARIN DAĞILIMI:")
    print("-------------------------------------------------------")
    
    # En yüksek olasılığa sahip olanı bulalım
    en_yuksek_skor_idx = np.argmax(reason_pred[0])
    en_olasi_neden = classes[en_yuksek_skor_idx]
    
    for idx, class_name in enumerate(classes):
        olasilik = reason_pred[0][idx] * 100
        # En yüksek olasılığı belirgin göstermek için yanına işaret koyalım
        isaret = " <--- [En Yüksek İhtimal]" if idx == en_yuksek_skor_idx else ""
        print(f"- {class_name.capitalize()}: %{olasilik:.2f}{isaret}")
    print("=======================================================")

if __name__ == "__main__":
    print("\n*** BEBEK SESİ TAHMİN SİSTEMİ ÇALIŞTI ***")
    # Kullanıcıdan dosya yolunu interaktif olarak alıyoruz
    dosya_yolu = input("\nLütfen test etmek istediğiniz .wav dosyasının tam yolunu yapıştırın:\n> ")
    
    # Kullanıcı tırnak içinde kopyaladıysa tırnakları temizleyelim
    dosya_yolu = dosya_yolu.strip('"').strip("'")
    
    ses_analiz_et(dosya_yolu)