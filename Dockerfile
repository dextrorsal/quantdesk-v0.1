# Multi-stage Dockerfile optimized for Railway deployment

# --- Build stage ---
FROM node:20-bookworm-slim AS build
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       g++ \
       make \
       python3 \
       git \
    && rm -rf /var/lib/apt/lists/*

# Copy root package.json first (for workspace setup)
COPY package*.json ./

# Copy backend package.json and install dependencies
COPY backend/package*.json ./backend/
WORKDIR /app/backend
RUN npm ci --only=production

# Copy backend source code
COPY backend/ ./

# Copy shared dependencies from root
WORKDIR /app
COPY tsconfig*.json ./

# --- Runtime stage ---
FROM node:20-bookworm-slim AS runner
WORKDIR /app

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built application
COPY --from=build /app ./

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
WORKDIR /app/backend
CMD ["node", "-r", "ts-node/register/transpile-only", "-r", "tsconfig-paths/register", "src/server.ts"]


