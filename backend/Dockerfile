FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy the entire backend directory AS a backend directory
COPY backend/ /app/backend/

# Copy your models and checkpoints
COPY models/ /app/models/
COPY checkpoints/ /app/checkpoints/

# Expose port (Documentation only)
EXPOSE 8000

# Run the API using the correct module path and Render's dynamic port
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]