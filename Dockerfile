# Simple Dockerfile for Railway deployment
FROM node:20-bookworm-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       g++ \
       make \
       python3 \
       git \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./
COPY backend/package*.json ./backend/

# Install dependencies
WORKDIR /app/backend
RUN npm ci --only=production

# Copy backend source code
COPY backend/ ./

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3002
ENV TS_NODE_TRANSPILE_ONLY=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3002/health || exit 1

EXPOSE 3002

# Start the application
CMD ["node", "-r", "ts-node/register/transpile-only", "-r", "tsconfig-paths/register", "src/server.ts"]