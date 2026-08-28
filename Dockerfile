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
# openai-whisper ships an sdist whose setup.py imports pkg_resources, which
# setuptools 81 removed. pip builds it in an isolated environment that pulls
# the newest setuptools, so the pin has to reach *that* env -- PIP_CONSTRAINT
# does; a line in requirements.txt would not.
RUN printf 'setuptools<81\n' > /tmp/pip-constraint.txt \
    && PIP_CONSTRAINT=/tmp/pip-constraint.txt pip install --no-cache-dir -r requirements.txt \
    && rm /tmp/pip-constraint.txt

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

# Fail the build, not the first scan. A packaged image that is missing the
# wordlists still starts and still accepts jobs -- it just kills every scan
# with an import error, which is how cleancut/data once shipped absent.
RUN python -c "\
from cleancut.config import Config; \
from cleancut.cli import main; \
import webapp.app as a; \
c = Config.load_defaults(); \
assert c.wordlists, 'wordlists.json missing from the image'; \
assert c.replacements, 'replacements.json missing from the image'; \
print('startup check ok: version', a.APP_VERSION, '|', len(c.wordlists), 'wordlist categories')"

EXPOSE 3000

CMD ["python", "-m", "webapp.app"]
