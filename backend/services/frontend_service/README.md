# Frontend Service Documentation

This document explains the complete lifecycle of the `app-frontend` and `app-backend` services from Docker build to runtime.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐         ┌─────────────────────┐            │
│  │    app-frontend     │         │     app-backend     │            │
│  │   (Next.js:3000)    │◄───────►│    (Flask:5000)     │            │
│  └──────────┬──────────┘         └──────────┬──────────┘            │
│             │                               │                        │
│             ▼                               ▼                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Persistent Volumes                            ││
│  │  frontend_repo:/workspace/ml-project                             ││
│  │  backend_repo:/workspace/ml-project                              ││
│  └─────────────────────────────────────────────────────────────────┘│
│             │                               │                        │
│             ▼                               ▼                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │              GitHub Repository (ml-project)                      ││
│  │              Branch: main (configurable)                         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Service Files Structure

```
services/frontend_service/
├── Dockerfile.app.frontend      # Frontend Docker image definition
├── Dockerfile.app.backend       # Backend Docker image definition
├── entrypoint_frontend.sh       # Frontend startup script
├── entrypoint_backend.sh        # Backend startup script
└── start_backend.py             # Python script to start Flask app
```

---

## 1. Docker Build Phase

### Frontend (`Dockerfile.app.frontend`)

```dockerfile
FROM node:lts-trixie-slim

WORKDIR /workspace

# Build arguments (passed from docker-compose.yml)
ARG GITHUB_TOKEN
ARG GITHUB_REPO=saadxelp/ml-project
ARG BRANCH=main

# Persist as environment variables for runtime
ENV GITHUB_TOKEN=$GITHUB_TOKEN
ENV GITHUB_REPO=$GITHUB_REPO
ENV BRANCH=$BRANCH

# Install git for cloning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and prepare entrypoint
COPY entrypoint_frontend.sh /workspace/entrypoint_frontend.sh
RUN chmod +x /workspace/entrypoint_frontend.sh

EXPOSE 3000

CMD ["/workspace/entrypoint_frontend.sh"]
```

**Key Points:**
- Base image: `node:lts-trixie-slim` (Node.js LTS)
- Installs only `git` - the actual code is cloned at runtime
- Build arguments are converted to environment variables
- The heavy work (clone + npm install) happens at **runtime**, not build time

### Backend (`Dockerfile.app.backend`)

```dockerfile
FROM python:3.12-slim

WORKDIR /workspace

ARG GITHUB_TOKEN
ARG GITHUB_REPO=saadxelp/ml-project
ARG BRANCH=main

ENV GITHUB_TOKEN=$GITHUB_TOKEN
ENV GITHUB_REPO=$GITHUB_REPO
ENV BRANCH=$BRANCH

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY start_backend.py /workspace/start_backend.py
COPY entrypoint_backend.sh /workspace/entrypoint_backend.sh
RUN chmod +x /workspace/start_backend.py /workspace/entrypoint_backend.sh

EXPOSE 5000

CMD ["/workspace/entrypoint_backend.sh"]
```

**Key Points:**
- Base image: `python:3.12-slim`
- Similar pattern to frontend
- Includes `start_backend.py` for Flask initialization

---

## 2. Container Startup Phase (Runtime)

When you run `docker compose up -d app-frontend app-backend`, the following happens:

### Step 1: Volume Mount

Docker creates/attaches persistent volumes:
```yaml
volumes:
  - frontend_repo:/workspace/ml-project  # For app-frontend
  - backend_repo:/workspace/ml-project   # For app-backend
```

These volumes **persist across container restarts**, saving clone and npm install time.

---

### Step 2: Entrypoint Execution (Frontend)

The `entrypoint_frontend.sh` script runs:

```bash
#!/bin/sh
set -e

REPO_DIR="/workspace/ml-project"

echo "Checking for repository..."

# CASE 1: Valid git repo exists (fastest path)
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repository exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull || echo "Git pull failed, continuing with existing code..."

# CASE 2: Directory has content but no .git (corrupted state)
elif [ -d "$REPO_DIR" ] && [ "$(ls -A $REPO_DIR 2>/dev/null)" ]; then
    echo "Directory has content but is not a git repo. Cleaning contents..."
    rm -rf "$REPO_DIR"/* "$REPO_DIR"/.[!.]* 2>/dev/null || true
    # Clone fresh
    git config --global http.version HTTP/1.1
    git clone --depth 1 -b $BRANCH https://$GITHUB_TOKEN@github.com/$GITHUB_REPO.git "$REPO_DIR"

# CASE 3: Empty directory (first run)
else
    echo "Cloning repository..."
    git config --global http.version HTTP/1.1
    git clone --depth 1 -b $BRANCH https://$GITHUB_TOKEN@github.com/$GITHUB_REPO.git "$REPO_DIR"
fi

cd "$REPO_DIR/frontend"

# Check if node_modules exists (skip npm install if already done)
if [ -d "node_modules" ] && [ "$(ls -A node_modules 2>/dev/null)" ]; then
    echo "Dependencies already installed. Skipping npm install..."
else
    echo "Installing dependencies..."
    npm install --legacy-peer-deps
fi

echo "Starting application..."
npm run dev
```

### Flowchart:

```
Container Start
      │
      ▼
┌─────────────────────┐
│ Check for .git dir  │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
  EXISTS      NOT EXISTS
     │           │
     ▼           ▼
┌─────────┐  ┌─────────────────┐
│git pull │  │ git clone       │
│ (fast)  │  │ (first run only)│
└────┬────┘  └────────┬────────┘
     │                │
     └───────┬────────┘
             ▼
┌───────────────────────────┐
│ Check node_modules exists │
└────────────┬──────────────┘
             │
      ┌──────┴──────┐
      │             │
   EXISTS       NOT EXISTS
      │             │
      ▼             ▼
┌──────────┐  ┌──────────────┐
│  SKIP    │  │ npm install  │
│ (fast)   │  │ (first run)  │
└────┬─────┘  └──────┬───────┘
     │               │
     └───────┬───────┘
             ▼
      ┌──────────────┐
      │ npm run dev  │
      │ (Next.js)    │
      └──────────────┘
```

---

## 3. Runtime Behavior

### Frontend (Next.js)
- Runs on port **3000** (exposed as **3001** on host)
- Development mode with hot-reload enabled
- Connects to `app-backend` for API calls

### Backend (Flask)
- Runs on port **5000** (exposed as **5000** on host)
- Connects to MongoDB for data storage
- Handles authentication (JWT), API endpoints

---

## 4. Performance Optimization Summary

| Scenario | What Happens | Time |
|----------|-------------|------|
| **First start** | Clone repo + npm install | ~5-10 min |
| **Container restart** | git pull + skip npm install | ~10-30 sec |
| **Container recreate** | Same as restart (volume persists) | ~10-30 sec |
| **Volume deleted** | Same as first start | ~5-10 min |
| **GitHub repo updated** | git pull fetches changes | ~10-30 sec |
| **package.json changed** | npm install runs again | ~2-5 min |

---

## 5. Commands Reference

### Start Services
```bash
docker compose up -d app-frontend app-backend
```

### View Logs
```bash
docker logs -f app-frontend
docker logs -f app-backend
```

### Restart (applies git changes)
```bash
docker compose restart app-frontend app-backend
```

### Full Rebuild (if entrypoint scripts change)
```bash
docker compose build --no-cache app-frontend app-backend
docker compose up -d --force-recreate app-frontend app-backend
```

### Clear Volumes (forces fresh clone/install)
```bash
docker compose down app-frontend app-backend
docker volume rm backend_frontend_repo backend_backend_repo
docker compose up -d app-frontend app-backend
```

---

## 6. Environment Variables

### Set in `.env` file:
```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxx    # Required for private repo access
GITHUB_REPO=saadxelp/ml-project  # Optional (has default)
BRANCH=main                       # Optional (has default)
```

### Used in containers:
| Variable | Used By | Purpose |
|----------|---------|---------|
| `GITHUB_TOKEN` | Both | Git authentication |
| `GITHUB_REPO` | Both | Repository to clone |
| `BRANCH` | Both | Git branch to use |
| `REDIS_HOST` | Both | Redis connection |
| `MONGO_URI` | Backend | MongoDB connection |
| `FLASK_ENV` | Backend | Flask environment |
| `JWT_SECRET_KEY` | Backend | JWT token signing |

---

## 7. Port Mappings

| Service | Container Port | Host Port | URL |
|---------|---------------|-----------|-----|
| app-frontend | 3000 | 3001 | http://localhost:3001 |
| app-backend | 5000 | 5000 | http://localhost:5000 |

---

## 8. Dependencies

### Frontend depends on:
- `redis`
- `loki`
- `grafana`
- `app-backend` (API server)

### Backend depends on:
- `redis`
- `mongodb` (via MONGO_URI)
- `loki`

---

## 9. Troubleshooting

### "GITHUB_TOKEN is not set"
- Ensure `.env` file exists with `GITHUB_TOKEN=your_token`

### "npm install takes forever on restart"
- Check if `node_modules` exists: `docker exec app-frontend ls -la /workspace/ml-project/frontend/node_modules`
- Volume might be corrupted - clear and restart

### "Cannot connect to backend"
- Check if backend is running: `docker ps | grep app-backend`
- Check backend logs: `docker logs app-backend`

### "git pull failed"
- Token might be expired
- Network issues
- Container continues with existing code (non-fatal)
