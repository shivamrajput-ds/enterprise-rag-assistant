FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=1200

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel packaging

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    --retries 30 \
    --timeout 1200 \
    --progress-bar off \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    --retries 30 \
    --timeout 1200 \
    --progress-bar off \
    -r requirements.txt

COPY . .

RUN mkdir -p data/documents data/vectorstore logs

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "start.sh"]