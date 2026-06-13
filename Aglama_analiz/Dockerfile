FROM python:3.11-slim

# librosa/soundfile için sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
