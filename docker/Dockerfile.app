FROM python:3.12-slim-bullseye AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy app-specific requirements file
COPY requirements-app.txt .

# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip config --user set global.progress_bar off
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-app.txt

# Final stage
FROM python:3.12-slim-bullseye

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
COPY app/ /app/app/
COPY models/ /app/models/

# Set environment variables
ENV PYTHONPATH=/app
ENV API_URL=http://api:8000

# Expose port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 