FROM python:3.11-slim

# ffmpeg does every mute, cut and subtitle burn; libsndfile backs librosa's
# audio loading; libgomp is torch's OpenMP runtime. libgl1/libglib2 are only
# needed if something in the tree resolves opencv-python instead of the headless
# build -- a few MB against `import cv2` failing at runtime, which the pipeline
# swallows as a warning and would silently skip the whole visual scan.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, installed first and separately. The default PyPI wheel drags
# in several gigabytes of CUDA libraries that an Umbrel Home can never use.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cleancut/ ./cleancut/
COPY webapp/ ./webapp/
COPY templates/ ./templates/
COPY static/ ./static/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=3000 \
    DATA_DIR=/app/data \
    MEDIA_ROOTS=/media/network:/media/home

RUN mkdir -p /app/data /media/network /media/home

EXPOSE 3000

CMD ["python", "-m", "webapp.app"]
