FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    torch \
    torchaudio \
    numpy \
    pretty_midi \
    librosa

COPY api.py .
COPY config.py .
COPY predict_on_custom.py .
COPY model/ ./model/
COPY results/ ./results/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]