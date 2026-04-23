# Secure Docker Build Instructions

## ✅ Security Improvements Made

The Dockerfiles have been refactored to **remove hardcoded GitHub tokens** for better security:

- ✅ Token is no longer visible in Docker image layers
- ✅ Token is passed as a build argument
- ✅ `.git` directory is removed after cloning
- ✅ `.env.build` file is gitignored

## 🔨 How to Build

### Option 1: Using .env.build file (Recommended)

1. **Load environment variables**:
   ```bash
   export $(cat services/frontend_service/.env.build | xargs)
   ```

2. **Build and start services**:
   ```bash
   docker compose build app-frontend app-backend
   docker compose up -d app-frontend app-backend
   ```

### Option 2: Pass token directly (One-time build)

```bash
GITHUB_TOKEN=ghp_YOUR_TOKEN docker compose build app-frontend app-backend
docker compose up -d app-frontend app-backend
```

### Option 3: Rebuild from scratch

```bash
# Load environment variables
export $(cat services/frontend_service/.env.build | xargs)

# Stop, rebuild (no cache), and start
docker compose stop app-frontend app-backend
docker compose build --no-cache app-frontend app-backend
docker compose up -d app-frontend app-backend
```

## 🔒 Security Best Practices

1. **Never commit `.env.build`** - Already added to `.gitignore`
2. **Rotate tokens regularly** - Update your GitHub Personal Access Token periodically
3. **Use minimal permissions** - Token should only have read access to the repository
4. **Restrict network access** - Use `127.0.0.1:PORT:PORT` in docker-compose for localhost-only access

## 📝 Environment Variables

### .env.build (Build-time secrets)
- `GITHUB_TOKEN` - GitHub Personal Access Token for cloning private repos
- `GITHUB_REPO` - Repository name (default: saadxelp/ml-project)
- `BRANCH` - Git branch to clone (default: main)

### .env (Runtime secrets)
- `MONGODB_URI` - MongoDB connection string for the application
- `JWT_SECRET_KEY` - JWT signing key
- etc.

## 🚨 Important Notes

- **Build arguments are not stored in the final image** if used correctly
- The `.git` directory is removed after cloning to prevent token leakage
- Keep your `.env.build` file secure and never share it
