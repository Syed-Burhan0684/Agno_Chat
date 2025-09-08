# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install minimal system deps needed for numeric libs and faiss
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libopenblas-dev \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for Docker layer caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# Create folder to persist indexes (the host mount will override this dir)
RUN mkdir -p /app/data/indexes

EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
