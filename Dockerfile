# Use official Python image with CUDA support
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Remove Python installation via apt and use deadsnakes PPA for stable Python
RUN apt-get update && apt-get install -y \
    curl \
    software-properties-common \
    libsndfile1 \
    libsamplerate0-dev \
    gcc \
    g++ \
    cmake \
    git \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Rest is same...
RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "python", "main.py"]