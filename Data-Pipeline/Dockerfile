# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# These are required for:
#  - ffmpeg (your existing code)
#  - matplotlib (libglib, libgl, freetype)
#  - seaborn
#  - numpy/scipy heavy ops
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Copy requirements
# --------------------------------------------------
COPY requirements-services.txt .
RUN pip install --no-cache-dir -r requirements-services.txt

# --------------------------------------------------
# MLflow + plotting + extra libraries
# --------------------------------------------------
RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    matplotlib \
    seaborn \
    numpy \
    pandas \
    boto3 \
    google-cloud-storage \
    gunicorn \
    waitress

# --------------------------------------------------
# Download SpaCy model
# --------------------------------------------------
RUN python -m spacy download en_core_web_trf

# --------------------------------------------------
# Copy the entire services directory
# --------------------------------------------------
COPY services/ services/

# --------------------------------------------------
# Copy start script
# --------------------------------------------------
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# --------------------------------------------------
# Expose ports
# --------------------------------------------------
EXPOSE 8000 8001

# --------------------------------------------------
# Env vars
# --------------------------------------------------
ENV OPENAI_API_KEY=""

# --------------------------------------------------
# Startup command
# --------------------------------------------------
CMD ["/app/start.sh"]