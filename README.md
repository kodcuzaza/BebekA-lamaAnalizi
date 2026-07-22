# Baby Cry Analysis - Docker Setup

## Running

1. Place the trained model files in the `models/` folder:
   - `models/bebek_uyku_modeli.h5`
   - `models/classes.npy`

   (These files are produced by running `ses_isleme.py`.)

2. Start the container:

```bash
   docker compose up --build
```

3. Open in browser: [http://localhost:8000](http://localhost:8000)

   The port can be changed via the `ports: "8000:5000"` line in `docker-compose.yml` (the left side is the externally accessible port).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web interface (audio file upload) |
| `POST /analyze` | Send a `.wav` file in the `file` field as form-data, returns a JSON result |
| `GET /health` | Health check |

## Notes

- `canli_test.py` (live listening via microphone) does not work inside the container, because it requires microphone access. The web service performs the same analysis via `app.py` by uploading an audio file.
- `ses_isleme.py` is for model training and is not included in the container (it requires a large dataset). Train the model on your own machine and copy it into the `models/` folder.
