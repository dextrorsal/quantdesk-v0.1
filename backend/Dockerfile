# Simple Dockerfile for backend directory deployment
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

# Copy backend files directly to /app
COPY . .

# Install npm-force-resolutions globally
RUN npm install -g npm-force-resolutions

# Install ALL dependencies (including dev dependencies for build)
RUN npm install

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV TS_NODE_TRANSPILE_ONLY=1

# Build the TypeScript project
RUN npm run build

# Remove dev dependencies after build
RUN npm prune --production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000

# Start the application
CMD ["node", "dist/server.js"]
