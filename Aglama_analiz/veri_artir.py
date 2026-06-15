import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

# Doğan gerçek dosya yolunu buraya sabitledik!
ANA_SES_KLASORU = r"C:\Users\USER\OneDrive\Desktop\Bebek_Uyku_Projesi\Baby Crying Sounds"
hedef_yol = os.path.join(ANA_SES_KLASORU, 'noise')

def gürültü_ekle(data, gürültü_orani=0.004):
    gürültü = np.random.randn(len(data))
    return (data + gürültü_orani * gürültü * np.max(data)).astype(np.float32)

def zamani_kaydir(data, kayma_orani=0.1):
    kayma_miktari = int(len(data) * np.random.uniform(-kayma_orani, kayma_orani))
    return np.roll(data, kayma_miktari)

print(f"🎯 Hedef Klasör: {hedef_yol}")

if os.path.exists(hedef_yol):
    # İçindeki tüm .ogg dosyalarını oku
    orijinal_sesler = [f for f in os.listdir(hedef_yol) if f.lower().endswith('.ogg')]
    
    print(f"✅ Klasör açıldı. İşlenecek {len(orijinal_sesler)} adet .ogg uzantılı gürültü sesiniz var.")
    toplam_yeni_ses = 0
    
    for dosya_adi in orijinal_sesler:
        ses_yolu = os.path.join(hedef_yol, dosya_adi)
        try:
            # Librosa ogg dosyasını okuyup 16kHz wav verisine hazırlar
            data, sr = librosa.load(ses_yolu, sr=16000, mono=True)
            ham_ad = os.path.splitext(dosya_adi)[0]
            
            # 1. Varyasyon: Gürültü eklenmiş .wav sürümü
            ses_gürültülü = gürültü_ekle(data)
            yolu_gürültü = os.path.join(hedef_yol, f"{ham_ad}_aug_noise.wav")
            wavfile.write(yolu_gürültü, sr, (ses_gürültülü * 32767).astype(np.int16))
            
            # 2. Varyasyon: Zamanı kaydırılmış .wav sürümü
            ses_kaymis = zamani_kaydir(data)
            yolu_kaymis = os.path.join(hedef_yol, f"{ham_ad}_aug_shift.wav")
            wavfile.write(yolu_kaymis, sr, (ses_kaymis * 32767).astype(np.int16))
            
            toplam_yeni_ses += 2
        except Exception as e:
            print(f"❌ Hata oluştu ({dosya_adi}): {e}")
            
    print(f"\n🎉 MÜKEMMEL! Noise klasörünün içine tam {toplam_yeni_ses} adet YENİ yapay .wav sesi eklendi.")
else:
    print("❌ HATA: Yol hâlâ yanlış, klasör bu konumda yok!")