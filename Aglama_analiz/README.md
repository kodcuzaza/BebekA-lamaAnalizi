# Bebek Ağlama Analizi - Docker Kurulumu

## Çalıştırma

1. Eğitilmiş model dosyalarını `models/` klasörüne koyun:
   - `models/bebek_uyku_modeli.h5`
   - `models/classes.npy`

   (Bu dosyalar `ses_isleme.py` çalıştırılarak üretilir.)

2. Container'ı başlatın:

   ```bash
   docker compose up --build
   ```

3. Tarayıcıda açın: http://localhost:8000

   Port `docker-compose.yml` içindeki `ports: "8000:5000"` satırından
   değiştirilebilir (sol taraf dışarıdan erişilecek port).

## API

- `GET /` -> Web arayüzü (ses dosyası yükleme)
- `POST /analyze` -> form-data ile `file` alanında .wav dosyası gönderilir,
  JSON sonuç döner.
- `GET /health` -> sağlık kontrolü

## Notlar

- `canli_test.py` (mikrofon ile canlı dinleme) container içinde çalışmaz,
  çünkü mikrofona erişim gerektirir. Web servisi `app.py` üzerinden ses
  dosyası yükleyerek aynı analizi yapar.
- `ses_isleme.py` model eğitimi içindir, container'a dahil edilmemiştir
  (büyük veri seti gerektirir). Modeli kendi makinenizde eğitip
  `models/` klasörüne kopyalayın.
