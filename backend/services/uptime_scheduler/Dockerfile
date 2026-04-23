# Lightweight Python image
FROM python:3.9-slim

# Install Docker CLI so we can run docker commands inside the container
# Also install curl to download docker client
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy files
COPY uptime_scheduler.py .
COPY .env .

# Set timezone (Optional, but good for scheduling)
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python dependencies (if we needed more than just standard lib, we'd use requirements.txt)
# But since we use manual env parsing now, we might not strictly need python-dotenv, 
# however, it's good practice to have it if we revert.
# RUN pip install python-dotenv 

# Run the scheduler
CMD ["python", "-u", "uptime_scheduler.py"]
