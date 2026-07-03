FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy requirements and install via uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy source code and artifacts
COPY backend/ ./backend/
COPY models/ ./models/
COPY .env .

# Environment variable for models path in Docker
ENV MODELS_DIR=/app/models

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
