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

MARKER_FILE="/workspace/.deps_installed"
REQ_FILE="$REPO_DIR/requirements.txt"

# Check if dependencies are already installed by comparing requirements hash
if [ -f "$MARKER_FILE" ] && [ -f "$REQ_FILE" ]; then
    CURRENT_HASH=$(md5sum "$REQ_FILE" | cut -d' ' -f1)
    SAVED_HASH=$(cat "$MARKER_FILE" 2>/dev/null)
    if [ "$CURRENT_HASH" = "$SAVED_HASH" ]; then
        echo "Dependencies already installed. Skipping pip install..."
    else
        echo "Requirements changed. Installing dependencies..."
        pip install -r "$REQ_FILE"
        echo "$CURRENT_HASH" > "$MARKER_FILE"
    fi
else
    echo "Installing dependencies..."
    pip install -r "$REQ_FILE"
    md5sum "$REQ_FILE" | cut -d' ' -f1 > "$MARKER_FILE"
fi

echo "Starting application..."
python /workspace/start_backend.py
