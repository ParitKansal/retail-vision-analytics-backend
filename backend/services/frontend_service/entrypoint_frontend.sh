#!/bin/sh
set -e

REPO_DIR="/workspace/ml-project"

echo "Checking for repository..."

# Check if .git exists (valid repo)
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repository exists. Pulling latest changes..."
    cd "$REPO_DIR"
    git pull || echo "Git pull failed, continuing with existing code..."
# Check if directory exists but has content (invalid state - clean it)
elif [ -d "$REPO_DIR" ] && [ "$(ls -A $REPO_DIR 2>/dev/null)" ]; then
    echo "Directory has content but is not a git repo. Cleaning contents..."
    # Remove contents but not the directory itself (since it's a mount point)
    rm -rf "$REPO_DIR"/* "$REPO_DIR"/.[!.]* 2>/dev/null || true
    echo "Cloning repository..."
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "Error: GITHUB_TOKEN is not set."
        exit 1
    fi
    git config --global http.version HTTP/1.1
    git clone --depth 1 -b $BRANCH https://$GITHUB_TOKEN@github.com/$GITHUB_REPO.git "$REPO_DIR"
else
    echo "Cloning repository..."
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "Error: GITHUB_TOKEN is not set."
        exit 1
    fi
    git config --global http.version HTTP/1.1
    git clone --depth 1 -b $BRANCH https://$GITHUB_TOKEN@github.com/$GITHUB_REPO.git "$REPO_DIR"
fi

cd "$REPO_DIR/frontend"

# Check if node_modules exists and has content
if [ -d "node_modules" ] && [ "$(ls -A node_modules 2>/dev/null)" ]; then
    echo "Dependencies already installed. Skipping npm install..."
else
    echo "Installing dependencies..."
    npm install --legacy-peer-deps
fi

echo "Starting application..."
npm run dev
