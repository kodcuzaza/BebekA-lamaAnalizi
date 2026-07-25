# 🍼 Akıllı Bebek Uyku Yönetimi — Yapay Zeka Destekli Bebek Sesi Analiz Sistemi

Bebek ses kayıtlarını analiz ederek **ağlama durumunu** ve olası **ağlama nedenini**
(açlık, karın ağrısı, yorgunluk, rahatsızlık vb.) tahmin eden, YAMNet tabanlı özellik
çıkarımı ve TensorFlow ile eğitilmiş çoklu görevli (multi-task) bir yapay zeka modeline
dayanan karar destek sistemi.

Bu proje, Bilecik Şeyh Edebali Üniversitesi İktisadi ve İdari Bilimler Fakültesi
Yönetim Bilişim Sistemleri Bölümü bitirme çalışması kapsamında geliştirilmiştir.

## ✨ Özellikler

- **Ağlama tespiti**: Bir ses kaydında ağlama olup olmadığını belirler (%99,92 doğruluk)
- **Neden tahmini**: Ağlama tespit edildiğinde olası nedeni tahmin eder (%62,35 doğruluk,
  9 sınıflı bir problemde rastgele tahminin ~%11 olduğu göz önüne alındığında yüksek bir başarı)
- **Web arayüzü**: Tarayıcıdan `.wav` dosyası yükleyip anlık sonuç alma
- **REST API**: `POST /analyze` ile programatik erişim
- **Docker desteği**: Tek komutla ayağa kaldırılabilir servis
- **Canlı mikrofon testi**: Yerel makinede mikrofon ile gerçek zamanlı deneme (`canli_test.py`)

## 🧠 Nasıl Çalışır?

1. Ses dosyası 16 kHz mono formata dönüştürülür (Librosa)
2. Önceden eğitilmiş **YAMNet** modeli ile 1024 boyutlu embedding (özellik vektörü) çıkarılır
3. Bu embedding, iki çıkışlı (multi-task) bir TensorFlow/Keras modeline verilir:
   - **cry_output**: Ağlama var mı? (sigmoid, ikili sınıflandırma)
   - **reason_output**: Ağlamanın olası nedeni nedir? (softmax, 9 sınıf)
4. Sonuçlar JSON olarak döner ve web arayüzünde çubuk grafiklerle görselleştirilir

Veri seti yaklaşık 1000 ses kaydından oluşmakta olup `noise` sınıfı Gaussian Noise
ekleme ve Time Shifting teknikleriyle (bkz. `veri_artir.py`) çoğaltılmıştır. Sınıflar:
`hungry`, `bellypain`, `tired`, `cold_hot`, `discomfort`, `burping`, `laugh`, `silence`,
`noise`.

> **Not:** Yukarıdaki sıralama sadece okunabilirlik içindir. `ses_isleme.py` çalıştığında
> `classes.npy` dosyasına yazılan gerçek sınıf sırası, `sklearn.LabelEncoder` tarafından
> **alfabetik** olarak belirlenir (`bellypain, burping, cold_hot, discomfort, hungry,
> laugh, noise, silence, tired`). Kodun hiçbir yerinde bu sıra sabit varsayılmaz —
> `app.py`, `test_et.py` ve `canli_test.py` her zaman `classes.npy` içeriğini okuyarak
> etiketleri eşler — ama modelin ham çıktısını (`reason_pred`) elle yorumlayacaksanız
> bu alfabetik sırayı bilmeniz gerekir.

## 📁 Proje Yapısı

```
.
├── app.py                 # Flask web servisi (ana uygulama)
├── ses_isleme.py           # Veri ön işleme + model eğitimi
├── veri_artir.py           # Veri artırma (data augmentation) betiği
├── canli_test.py           # Mikrofon ile canlı test (yerel kullanım)
├── test_et.py               # Tek bir .wav dosyasını komut satırından test etme
├── templates/
│   └── index.html          # Web arayüzü
├── models/                 # Eğitilmiş model dosyaları buraya konur (repoya dahil değildir)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── LICENSE
```

## 🚀 Kurulum ve Çalıştırma

### Seçenek 1 — Docker (önerilen)

1. Eğitilmiş model dosyalarını `models/` klasörüne yerleştirin:
   - `models/bebek_uyku_modeli.h5`
   - `models/classes.npy`

   (Bu dosyalar `ses_isleme.py` çalıştırılarak üretilir, bkz. [Model Eğitimi](#-model-eğitimi).)

2. Container'ı başlatın:

   ```bash
   docker compose up --build
   ```

3. Tarayıcıda açın: [http://localhost:8000](http://localhost:8000)

   Port, `docker-compose.yml` içindeki `ports: "8000:5000"` satırından değiştirilebilir
   (sol taraf dışarıdan erişilecek port).

### Seçenek 2 — Yerel Python ortamı

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Model dosyalarını proje köküne koyun (veya MODEL_YOLU / SINIF_YOLU ile yol belirtin)
python app.py
```

## 🔌 API

| Endpoint | Açıklama |
|----------|-------------|
| `GET /` | Web arayüzü (ses dosyası yükleme) |
| `POST /analyze` | `file` alanında form-data ile `.wav` dosyası gönderilir, JSON sonuç döner |
| `GET /health` | Sağlık kontrolü |

Örnek `/analyze` yanıtı:

```json
{
  "is_crying": true,
  "crying_probability": 0.9996,
  "top_reason": "bellypain",
  "probabilities": [
    { "label": "bellypain", "probability": 0.4598 },
    { "label": "tired", "probability": 0.2946 }
  ]
}
```

## 🏋️ Model Eğitimi

Model, repoya dahil edilmemiştir (büyük bir veri seti gerektirir). Kendi veri setinizle
yeniden eğitmek için:

1. Veri setinizi aşağıdaki klasör yapısında hazırlayın:

   ```
   dataset/
   ├── hungry/*.wav
   ├── bellypain/*.wav
   ├── tired/*.wav
   ├── cold_hot/*.wav
   ├── discomfort/*.wav
   ├── burping/*.wav
   ├── laugh/*.wav
   ├── silence/*.wav
   └── noise/*.wav
   ```

2. (İsteğe bağlı) `noise` sınıfını çoğaltmak isterseniz:

   ```bash
   export DATASET_PATH=/path/to/dataset
   python veri_artir.py
   ```

3. Modeli eğitin:

   ```bash
   export DATASET_PATH=/path/to/dataset
   python ses_isleme.py
   ```

   Bu işlem sonunda `bebek_uyku_modeli.h5` ve `classes.npy` dosyaları üretilir.
   Bu dosyaları `models/` klasörüne kopyalayıp web servisini başlatabilirsiniz.

## 📊 Sonuçlar

| Görev | Doğruluk |
|-------|----------|
| Ağlama tespiti (`cry_output_accuracy`) | %99,92 |
| Ağlama nedeni tahmini (`reason_output_accuracy`) | %62,35 |

Model 30 epoch boyunca, %80 eğitim / %20 test veri bölünmesiyle eğitilmiştir.

## ⚠️ Notlar ve Sınırlamalar

- `canli_test.py` mikrofon erişimi gerektirdiğinden Docker container'ı içinde çalışmaz;
  yalnızca yerel makinede kullanılabilir. Web servisi (`app.py`) aynı analizi ses dosyası
  yükleyerek gerçekleştirir. Bu betiği çalıştırmadan önce, `requirements.txt`'e dahil
  olmayan `sounddevice` paketini ayrıca kurmanız gerekir: `pip install sounddevice`
  (macOS/Linux'ta ayrıca PortAudio sistem kütüphanesi gerekebilir).
- Bu sistem doğrudan tıbbi teşhis koymayı amaçlamaz; ebeveynlere karar destek sağlayan
  bir ön değerlendirme aracıdır.

## 🔮 Gelecek Planları

- Mobil uygulama entegrasyonu
- IoT cihaz desteği (akıllı bebek monitörleri)
- Gerçek zamanlı bildirim sistemi
- Bulut tabanlı veri depolama
- Uyku takibi ve günlük bakım raporları

## 🛠️ Kullanılan Teknolojiler

- **Python**, **TensorFlow** / **TensorFlow Hub** (YAMNet)
- **Librosa** (ses işleme)
- **Flask** + **Gunicorn** (web servisi)
- **Docker** / **Docker Compose**

## 👥 Geliştiriciler

- Batuhan Göçmen
- Muhammed Ali Yetimçok

**Danışman:** Dr. Öğr. Üyesi Hüseyin Parmaksız
Bilecik Şeyh Edebali Üniversitesi — İktisadi ve İdari Bilimler Fakültesi
Yönetim Bilişim Sistemleri Bölümü, 2026

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
