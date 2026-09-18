FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860 \
    TORCH_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /home/app/project
COPY requirements.txt ./requirements.txt
RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY --chown=app:app . .
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/config', timeout=8)"
CMD ["python", "deployment.py"]
