# FROM python:3.10-slim
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    && apt-get clean

# Use python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy app code
WORKDIR /app
COPY . /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --verbose

# Expose the port your app runs on
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "remy_server_v1:app", "--host", "0.0.0.0", "--port", "8000"]
