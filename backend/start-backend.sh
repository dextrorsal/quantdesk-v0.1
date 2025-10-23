#!/bin/bash

# QuantDesk Backend Startup Script with Redis Support
# Usage: ./start-backend.sh [--redis] [--help]
#   --redis: Start Redis in Docker container
#   --help:  Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
START_REDIS=false
REDIS_CONTAINER_NAME="quantdesk-redis"
REDIS_PORT="6379"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --redis)
            START_REDIS=true
            shift
            ;;
        --help|-h)
            echo "QuantDesk Backend Startup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --redis    Start Redis in Docker container"
            echo "  --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Start backend only"
            echo "  $0 --redis        # Start backend with Redis"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
        echo -e "${YELLOW}Please install Docker to use the --redis flag${NC}"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker daemon is not running${NC}"
        echo -e "${YELLOW}Please start Docker daemon to use the --redis flag${NC}"
        exit 1
    fi
}

# Function to start Redis
start_redis() {
    echo -e "${BLUE}🔍 Checking Redis status...${NC}"
    
    # Check if port 6379 is already in use
    if docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -q ":${REDIS_PORT}->"; then
        echo -e "${YELLOW}⚠️ Port ${REDIS_PORT} is already in use by another container${NC}"
        
        # Find which container is using the port
        PORT_CONTAINER=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep ":${REDIS_PORT}->" | awk '{print $1}')
        echo -e "${BLUE}Found container '${PORT_CONTAINER}' using port ${REDIS_PORT}${NC}"
        
        # Ask if we should stop it or use it
        if [ "$PORT_CONTAINER" = "${REDIS_CONTAINER_NAME}" ]; then
            echo -e "${GREEN}✅ Our Redis container is already running${NC}"
        else
            echo -e "${YELLOW}🔄 Stopping conflicting container '${PORT_CONTAINER}'...${NC}"
            docker stop ${PORT_CONTAINER} 2>/dev/null || true
            docker rm ${PORT_CONTAINER} 2>/dev/null || true
            sleep 2
        fi
    fi
    
    # Check if our Redis container already exists
    if docker ps -a --format "table {{.Names}}" | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        # Container exists, check if it's running
        if docker ps --format "table {{.Names}}" | grep -q "^${REDIS_CONTAINER_NAME}$"; then
            echo -e "${GREEN}✅ Redis container is already running${NC}"
        else
            echo -e "${YELLOW}🔄 Starting existing Redis container...${NC}"
            docker start ${REDIS_CONTAINER_NAME}
            sleep 2
        fi
    else
        echo -e "${BLUE}🚀 Creating new Redis container...${NC}"
        docker run -d \
            --name ${REDIS_CONTAINER_NAME} \
            -p ${REDIS_PORT}:6379 \
            redis:alpine \
            redis-server --appendonly yes
        sleep 3
    fi
    
    # Test Redis connection
    echo -e "${BLUE}🔍 Testing Redis connection...${NC}"
    if docker exec ${REDIS_CONTAINER_NAME} redis-cli ping | grep -q "PONG"; then
        echo -e "${GREEN}✅ Redis is ready and responding${NC}"
    else
        echo -e "${RED}❌ Redis is not responding properly${NC}"
        exit 1
    fi
}

# Function to stop Redis
stop_redis() {
    if docker ps --format "table {{.Names}}" | grep -q "^${REDIS_CONTAINER_NAME}$"; then
        echo -e "${YELLOW}🛑 Stopping Redis container...${NC}"
        docker stop ${REDIS_CONTAINER_NAME}
    fi
}

# Function to cleanup processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down backend...${NC}"
    
    # Kill nodemon and related processes
    pkill -f "nodemon" 2>/dev/null || true
    pkill -f "ts-node" 2>/dev/null || true
    pkill -f "node.*server.ts" 2>/dev/null || true
    
    # Kill any process on port 3002
    if lsof -ti:3002 >/dev/null 2>&1; then
        echo -e "${YELLOW}Killing process on port 3002...${NC}"
        lsof -ti:3002 | xargs kill -9 2>/dev/null || true
    fi
    
    # Stop Redis if we started it
    if [ "$START_REDIS" = true ]; then
        stop_redis
    fi
    
    echo -e "${GREEN}✅ Backend stopped cleanly${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🚀 Starting QuantDesk Backend...${NC}"

# Start Redis if requested
if [ "$START_REDIS" = true ]; then
    check_docker
    start_redis
    echo -e "${GREEN}✅ Redis is ready${NC}"
fi

echo -e "${YELLOW}Press Ctrl+C to stop${NC}"

# Change to backend directory
cd "$(dirname "$0")"

# Start the backend with nodemon
TS_NODE_TRANSPILE_ONLY=1 npx nodemon --exec "node -r ts-node/register" src/server.ts

# If we get here, nodemon exited
cleanup
