FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Cloud Run uses PORT environment variable, default to 8080
ENV PORT=8501

# Expose the port
EXPOSE 8501

# Run Streamlit on the PORT that Cloud Run provides
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.maxUploadSize=2000