FROM python:3.9-slim

# Create a non-root user
RUN useradd -m appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY service.py .

# Ensure the appuser owns the directory (to write last_ip.txt)
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

CMD ["python", "service.py"]
