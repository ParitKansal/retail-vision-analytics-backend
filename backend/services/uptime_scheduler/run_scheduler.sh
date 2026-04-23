#!/bin/bash

# Navigate to the script directory
cd "$(dirname "$0")"

echo "Building and Starting Uptime Scheduler Container..."

# Run docker-compose for the scheduler
# -d runs in detached mode (background)
# --build ensures we rebuild if the python script changes
docker-compose -p scheduler_stack up --build -d

echo "Scheduler is running in background container 'uptime_scheduler'."
echo "View logs with: docker logs -f uptime_scheduler"
