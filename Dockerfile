# Railway Dockerfile - handle missing backend gracefully
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

# Try to install dependencies if backend exists
RUN if [ -d "backend" ]; then \
        cd backend && npm install --only=production; \
    else \
        echo "Backend directory not found, skipping npm install"; \
    fi

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