FROM node:20-bookworm-slim

WORKDIR /app

# Copy backend files
COPY backend/ .

# Install all dependencies (including dev dependencies for ts-node)
RUN npm install

# Set environment
ENV NODE_ENV=production
ENV PORT=3002

# Start application
CMD ["node", "-r", "ts-node/register/transpile-only", "src/server.ts"]