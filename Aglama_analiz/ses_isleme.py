import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import scipy.signal
import librosa  # Yeni eklenen akıllı ses kütüphanesi

# 1. Dataset Yolunu Tanımlayalım
DATASET_PATH = r"C:\Users\USER\OneDrive\Desktop\Bebek_Uyku_Projesi\Baby Crying Sounds"

# 2. YAMNet Modelini TensorFlow Hub'dan Yükleyelim
print("YAMNet yükleniyor, lütfen bekleyin...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# 3. Ses dosyalarını hatasız şekilde okuyan ve 16kHz formatına dönüştüren fonksiyon
def load_and_resample_audio(file_path):
    # Librosa dosyayı otomatik olarak 16000Hz ve Mono (tek kanal) olarak yükler
    # Format 3 (Float) veya gizli m4a/wav karmaşası librosa için sorun yaratmaz
    audio_numpy, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # YAMNet'e göndermeden önce TensorFlow tensörüne çeviriyoruz
    audio = tf.convert_to_tensor(audio_numpy, dtype=tf.float32)
    return audio

# 4. Klasörleri ve Etiket Mantığını Tanımlayalım
cry_classes = ['hungry', 'bellypain', 'tired', 'cold_hot', 'discomfort', 'burping']
no_cry_classes = ['silence', 'laugh', 'noise']
all_classes = cry_classes + no_cry_classes

# Verileri toplayacağımız listeler
X_embeddings = []  # YAMNet'ten çıkan 1024'lük sayılar buraya gelecek
y_reason = []      # Ağlama nedeni (hungry, tired vb.) buraya gelecek
y_is_crying = []   # Ağlama var mı? (1: Evet, 0: Hayır) buraya gelecek

print("\nSes dosyaları işleniyor ve embedding'ler çıkarılıyor...")

# 5. Tüm Klasörleri Tarayalım
for class_name in all_classes:
    class_dir = os.path.join(DATASET_PATH, class_name)
    
    # Eğer klasör bilgisayarda yoksa atla (hata vermemesi için)
    if not os.path.exists(class_dir):
        print(f"Uyarı: {class_name} klasörü bulunamadı, atlanıyor.")
        continue
        
    # Klasörün içindeki .wav dosyalarını bul
    file_names = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    print(f"- {class_name} klasörü işleniyor ({len(file_names)} dosya)...")
    
    for file_name in file_names:
        file_path = os.path.join(class_dir, file_name)
        
        try:
            # Sesi yükle ve hazırla (Artık Librosa devrede)
            wav_data = load_and_resample_audio(file_path)
            
            # YAMNet modeline sesi gönderip embedding alalım
            scores, embeddings, spectrogram = yamnet_model(wav_data)
            
            if embeddings.shape[0] > 0:
                # YAMNet ses boyunca birden fazla embedding üretebilir, ortalamasını alarak tek bir imza elde ediyoruz
                mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
                
                # Listelerimize ekleyelim
                X_embeddings.append(mean_embedding) # 1024 sayı
                y_reason.append(class_name)         # Örn: 'hungry'
                
                # Ağlama durumunu belirleyelim (1 veya 0)
                is_cry_label = 1 if class_name in cry_classes else 0
                y_is_crying.append(is_cry_label)
                
        except Exception as e:
            # Hatalı veya bozuk bir ses dosyası olursa programın çökmesini engeller
            print(f"Hata oluştu ({file_name}): {str(e)}")

# numpy dizilerine dönüştürelim
X_embeddings = np.array(X_embeddings)
y_reason = np.array(y_reason)
y_is_crying = np.array(y_is_crying)

print("\n--- VERİ ÖN İŞLEME TAMAMLANDI ---")
print(f"Toplam başarıyla işlenen ses sayısı: {X_embeddings.shape[0]}")
print(f"Embedding matris boyutu: {X_embeddings.shape} (Her ses için 1024 özellik)")

# =====================================================================
# MODÜL 2: YAPAY ZEKA MODELİNİN KURULMASI VE EEĞİTİLMESİ
# =====================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input

print("\n--> Yapay Zeka Eğitim Aşaması Başlıyor...")

# 1. Klasör isimlerini (hungry, tired vb.) yapay zekanın anlayacağı sayılara (0, 1, 2...) dönüştürelim
label_encoder = LabelEncoder()
y_reason_encoded = label_encoder.fit_transform(y_reason)
num_classes = len(label_encoder.classes_)

# 2. Veri Setini Eğitim (%80) ve Test (%20) olarak ikiye bölelim
X_train, X_test, y_cry_train, y_cry_test, y_reason_train, y_reason_test = train_test_split(
    X_embeddings, 
    y_is_crying, 
    y_reason_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_reason_encoded
)

# 3. Çoklu Görev (Multi-task) Model Mimarisinin Tasarlanması
input_layer = Input(shape=(1024,), name="yamnet_input")

# Ortak Katmanlar (Sesin genel özelliklerini öğrenen ana beyin)
shared = Dense(256, activation='relu')(input_layer)
shared = Dropout(0.3)(shared)
shared = Dense(128, activation='relu')(shared)
shared = Dropout(0.3)(shared)

# Çıktı Kolu 1: Ağlama Var mı Yok mu? (Evet/Hayır için Sigmoid)
output_cry = Dense(1, activation='sigmoid', name='cry_output')(shared)

# Çıktı Kolu 2: Sesin Nedeni Nedir? (Çoklu sınıf için Softmax)
output_reason = Dense(num_classes, activation='softmax', name='reason_output')(shared)

# Modeli birleştirelim
model = Model(inputs=input_layer, outputs=[output_cry, output_reason])

# 4. Modelin Derlenmesi
model.compile(
    optimizer='adam',
    loss={
        'cry_output': 'binary_crossentropy',
        'reason_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'cry_output': 'accuracy',
        'reason_output': 'accuracy'
    }
)

# Modelin yapısını ekrana yazdıralım
model.summary()

# 5. Modelin Eğitilmesi (30 tur/epoch boyunca öğrenme gerçekleşecek)
print("\nModel eğitiliyor, lütfen bekleyin (Epochs)...")
history = model.fit(
    X_train, 
    {'cry_output': y_cry_train, 'reason_output': y_reason_train},
    validation_data=(X_test, {'cry_output': y_cry_test, 'reason_output': y_reason_test}),
    epochs=30,
    batch_size=32
)

# 6. Eğitilen Modelin ve Etiket Düzeninin Bilgisayara Kaydedilmesi
model.save("bebek_uyku_modeli.h5")
np.save("classes.npy", label_encoder.classes_)

print("\n=======================================================")
print("TEBRİKLER! Model başarıyla eğitildi ve kaydedildi.")
print("- Yapay Zeka Modeli: 'bebek_uyku_modeli.h5' olarak kaydedildi.")
print("- Sınıf Etiketleri: 'classes.npy' olarak kaydedildi.")
print("=======================================================")