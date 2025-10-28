# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy the entire services directory
COPY services/ services/

# Copy schema to text_processing
# RUN cp services/text_processing/schema.sql services/text_processing/

# Create a startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 8001

# Set environment variable for OpenAI key (to be passed at runtime)
ENV OPENAI_API_KEY=""

# Run the startup script
CMD ["/app/start.sh"]