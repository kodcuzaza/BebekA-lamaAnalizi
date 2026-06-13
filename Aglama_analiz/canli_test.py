import os
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

# 1. Gerekli Ayarlar ve Dosya Yolları
MODEL_YOLU = "bebek_uyku_modeli.h5"
SINIF_YOLU = "classes.npy"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

print("🧠 Yapay zeka beyni ve YAMNet yükleniyor, lütfen bekleyin...")
model = tf.keras.models.load_model(MODEL_YOLU, custom_objects={'KerasLayer': hub.KerasLayer})
classes = np.load(SINIF_YOLU, allow_pickle=True)
yamnet_model = hub.load(YAMNET_URL)

# Canlı ses alma ayarları
SURE = 5  # Her tahminde kaç saniyelik ses dinlenecek?
SAMPLING_RATE = 16000  # YAMNet 16kHz ses ister

def embedding_cikar(audio_data):
    """Mikrofondan alınan sesi YAMNet formatına sokar ve özellik çıkarır."""
    # Sesi [-1.0, 1.0] arasına normalize et
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    _, embeddings, _ = yamnet_model(audio_data)
    # Eğitimdeki gibi ortalamasını alarak tek bir vektör (1024,) elde et
    return np.mean(embeddings.numpy(), axis=0)

print("\n🎙️ --- CANLI TAHMİN SİSTEMİ HAZIR ---")
print(f"Sistem her {SURE} saniyede bir ortamı dinleyip analiz edecek.")
print("Durdurmak için terminalde Ctrl + C tuşlarına basabilirsiniz.\n")

try:
    while True:
        print("🎧 Dinleniyor... (Ortamda ses çıkarın veya bebek ağlaması açın)")
        
        # Mikrofondan 5 saniyelik ses kaydet
        kayit = sd.rec(int(SURE * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='float32')
        sd.wait()  # Kayıt bitene kadar bekle
        
        # Sesi düzleştir (1D Array yap)
        ses_verisi = kayit.flatten()
        
        # Özellikleri çıkar
        features = embedding_cikar(ses_verisi)
        features = np.expand_dims(features, axis=0) # Modelin istediği (1, 1024) boyutuna getir
        
        # Yapay zekaya tahmin ettir
        cry_pred, reason_pred = model.predict(features, verbose=0)
        
        # Sonuçları yorumla
        is_crying = cry_pred[0][0] > 0.5  # %50'den büyükse ağlama kabul et
        reason_index = np.argmax(reason_pred[0])
        reason_label = classes[reason_index]
        reason_conf = reason_pred[0][reason_index] * 100
        
        print("-" * 40)
        if is_crying:
            print(f"🚨 DURUM: Bebek Ağlıyor! (Eminlik: %{cry_pred[0][0]*100:.2f})")
            print(f"📌 TAHMİNİ NEDEN: {reason_label.upper()} (Olasılık: %{reason_conf:.2f})")
        else:
            # Eğer gürültü sınıfı baskın çıktıysa veya ağlama algılanmadıysa
            if reason_label == 'noise':
                print(f"🍃 DURUM: Ağlama yok, sadece ortam gürültüsü/dış ses var.")
            elif reason_label == 'silence':
                print(f"🤫 DURUM: Ortam sessiz.")
            else:
                print(f"✅ DURUM: Ağlama algılanmadı. Duyulan ses: {reason_label.upper()}")
        print("-" * 40 + "\n")

except KeyboardInterrupt:
    print("\n👋 Canlı test sonlandırıldı. Harika bir iş çıkardın!")