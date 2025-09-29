# Railway Dockerfile - handle npm-force-resolutions
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

# Copy everything
COPY . .

# Debug: show what was copied
RUN ls -la
RUN ls -la backend/ || echo "Backend directory not found"

# Install dependencies in backend directory
WORKDIR /app/backend

# Install npm-force-resolutions globally first
RUN npm install -g npm-force-resolutions

# Then install project dependencies
RUN npm install --only=production

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

# Default command (Railway will override with startCommand)
CMD ["echo", "Container ready - Railway will use startCommand"]