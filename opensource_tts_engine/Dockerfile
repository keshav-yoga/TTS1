
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ffmpeg libsndfile1         && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip &&         pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "tts_engine.api.server"]
