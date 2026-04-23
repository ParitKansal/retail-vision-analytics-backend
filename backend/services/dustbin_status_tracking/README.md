# Dustbin Status Tracking Service

## Overview
The `dustbin_status_tracking` service is a microservice that monitors dustbin fill levels using computer vision. It processes camera frames in real-time and detects three states:
- **operational**: Dustbin is at normal fill level
- **undetectable**: Person is blocking the view (cannot determine status)
- **full**: Dustbin is full and needs emptying

## Architecture
This service follows the standard microservice pattern used throughout the backend:
- Consumes frames from Redis streams (published by `stream_handling_service`)
- Processes frames using computer vision algorithms
- Publishes results to output queue for database storage
- Integrates with Loki for centralized logging

## Key Features
- **Redis Stream Integration**: Reads from input queue, writes to output queue
- **Loki Logging**: All events logged to Grafana Loki
- **Smart Detection**: Uses pixel sum thresholds and contour analysis
- **Status Smoothing**: Uses history-based majority voting to avoid flicker
- **Configurable Regions**: Polygon regions defined via environment variables

## Configuration

### Environment Variables (.env)
The service is configured through environment variables:

**Redis & Logging:**
- `REDIS_HOST`: Redis server hostname (default: "redis")
- `REDIS_PORT`: Redis server port (default: 6379)
- `LOKI_URL`: Loki push endpoint for logs
- `INPUT_QUEUE_NAME`: Redis stream to consume frames from
- `OUTPUT_QUEUE_NAME_1`: Redis stream to publish results
- `GROUP_NAME`: Consumer group name for coordinated processing

**Detection Parameters:**
- `REGION_POLYGON`: JSON array of polygon points defining the detection region
- `LOWER_THRESHOLD`: Lower pixel sum threshold (default: 600000)
- `UPPER_THRESHOLD`: Upper pixel sum threshold (default: 750000)
- `DARK_PATCH_SENSITIVITY`: Sensitivity for person detection (default: 30)
- `LOCAL_DEVIATION_THRESHOLD`: Threshold for local strip check (default: 20)
- `LOCAL_STRIP_THRESHOLD_PERCENT`: Percentage for strip threshold (default: 40.0)
- `MIN_CONTOUR_AREA`: Minimum contour area to consider (default: 300)
- `EDGE_MARGIN`: Edge margin for connectivity check (default: 5)
- `MORPH_KERNEL_SIZE`: Morphological operation kernel size (default: 3)

## Adding to Docker Compose

To add this service to your `docker-compose.yml`, add the following configuration:

```yaml
  dustbin_status_tracking:
    build: ./services/dustbin_status_tracking
    container_name: dustbin_status_tracking
    environment:
      - REDIS_HOST=redis
      - LOKI_URL=http://loki:3100/loki/api/v1/push
      - TZ=Asia/Kolkata
      - INPUT_QUEUE_NAME=stream:cam:5
      - GROUP_NAME=cg:cam:dustbin_status
      # Optional: Override detection parameters
      # - LOWER_THRESHOLD=600000
      # - UPPER_THRESHOLD=750000
    networks:
      - app_network
    depends_on:
      - redis
      - loki
      - grafana
      - db_service
      - stream_handling_service
      - mongodb
```

**Note**: Adjust `INPUT_QUEUE_NAME` to match the camera stream you want to monitor.

## How It Works

1. **Frame Reception**: Service reads frames from Redis stream (published by `stream_handling_service`)
2. **Region Extraction**: Extracts the configured polygon region from the frame
3. **Pixel Sum Analysis**: Calculates total RGB pixel sum in the region
4. **Threshold Check**: If within thresholds → status is "operational"
5. **Person Detection** (if outside thresholds):
   - **Vertical Connectivity Check**: Detects person-shaped dark patches spanning the region
   - **Local Strip Check**: Detects significant deviation in bottom-half strip
   - If person detected → status is "undetectable"
   - If no person → status is "full"
6. **Status Smoothing**: Uses 5-frame history with majority voting
7. **Result Publishing**: Publishes status to output queue for database storage

## Dependencies
- `redis==7.1.0`: Redis client for stream processing
- `requests==2.32.5`: HTTP client for downloading frames
- `python-dotenv==1.2.1`: Environment variable management
- `python-logging-loki==0.3.1`: Loki logging integration
- `opencv-python-headless==4.8.1.78`: Computer vision operations
- `numpy==1.26.2`: Numerical operations
- `Pillow==10.1.0`: Image processing

## Testing

### Local Testing
To test locally (without Docker):
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export REDIS_HOST=localhost
export INPUT_QUEUE_NAME=stream:cam:5
export GROUP_NAME=cg:cam:dustbin_status

# Run service
python service.py
```

### Docker Testing
```bash
# Build the image
docker build -t dustbin_status_tracking ./services/dustbin_status_tracking

# Run container
docker run --network app_network dustbin_status_tracking
```

## Output Format

The service publishes JSON messages to the output queue:

```json
{
  "camera_id": "cam_5",
  "timestamp": "2025-12-10T16:45:00",
  "status": "operational",
  "target_collection": "dustbin_status_collection"
}
```

## Monitoring

View logs in Grafana by filtering for:
- **Application**: `dustbin_status_tracking`
- **Log Level**: INFO, ERROR

Key log messages:
- `DustbinStatusDetector initialized with ROI: ...` - Service startup
- `Published dustbin status for camera X at Y: Z` - Successful detection
- `Failed to download frame from ...` - Frame download error

## Comparison with crew_room_service

This service follows the same architectural pattern as `crew_room_service`:

**Similarities:**
- ✅ Redis stream consumer/producer
- ✅ Loki logging integration
- ✅ Environment-based configuration
- ✅ Docker containerization
- ✅ Continuous processing loop
- ✅ Error handling and retry logic

**Differences:**
- Uses custom computer vision algorithms instead of YOLO
- More configuration parameters for detection tuning
- Status smoothing with history-based voting
- Downloads frames from URLs (MinIO) instead of direct frame data

## Future Enhancements

Potential improvements:
1. Add support for multiple dustbin regions per camera
2. Implement adaptive threshold calibration
3. Add historical trend analysis
4. Integrate with notification service for "full" alerts
5. Add visual debugging mode with annotated frames
