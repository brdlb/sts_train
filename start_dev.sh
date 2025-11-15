#!/bin/bash

# Script to start backend server, frontend, and SSH tunnel
# Usage: ./start_dev.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REMOTE_HOST="46.101.158.230"
REMOTE_PORT="5565"
LOCAL_SERVER_PORT="5565"
LOCAL_FRONTEND_PORT="5174"
SSH_TUNNEL_PORT="5565"
# SSH user (can be set via SSH_USER environment variable)
SSH_USER="${SSH_USER:-}"
SSH_TARGET="${SSH_USER:+${SSH_USER}@}${REMOTE_HOST}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    
    # Kill backend server
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    # Kill frontend
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Kill SSH tunnel
    if [ ! -z "$SSH_PID" ]; then
        echo "Stopping SSH tunnel (PID: $SSH_PID)..."
        kill $SSH_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup SIGINT SIGTERM

# Check if ports are already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an | grep -q ":$port.*LISTEN" 2>/dev/null; then
        echo -e "${RED}Error: Port $port is already in use${NC}"
        exit 1
    fi
}

echo -e "${GREEN}Starting development environment...${NC}"

# Check ports
echo "Checking ports..."
check_port $LOCAL_SERVER_PORT
check_port $LOCAL_FRONTEND_PORT

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Start backend server
echo -e "\n${GREEN}Starting backend server on 127.0.0.1:$LOCAL_SERVER_PORT...${NC}"
python -m src.perudo.web.main > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend server started (PID: $BACKEND_PID)"

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend server failed to start${NC}"
    cat backend.log
    exit 1
fi

# Start frontend
echo -e "\n${GREEN}Starting frontend on localhost:$LOCAL_FRONTEND_PORT...${NC}"
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend started (PID: $FRONTEND_PID)"

# Wait for frontend to start
sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start${NC}"
    cat frontend.log
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start SSH tunnel
echo -e "\n${GREEN}Starting SSH tunnel to $SSH_TARGET:$REMOTE_PORT...${NC}"
echo "SSH tunnel will forward remote port $REMOTE_PORT to local port $SSH_TUNNEL_PORT"
ssh -N -L $SSH_TUNNEL_PORT:127.0.0.1:$REMOTE_PORT $SSH_TARGET > ssh_tunnel.log 2>&1 &
SSH_PID=$!
echo "SSH tunnel started (PID: $SSH_PID)"

# Wait for SSH tunnel to establish
sleep 2

# Check if SSH tunnel is running
if ! kill -0 $SSH_PID 2>/dev/null; then
    echo -e "${YELLOW}Warning: SSH tunnel may have failed to start${NC}"
    cat ssh_tunnel.log
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Backend server: ${GREEN}http://127.0.0.1:$LOCAL_SERVER_PORT${NC}"
echo -e "Frontend:       ${GREEN}http://localhost:$LOCAL_FRONTEND_PORT${NC}"
echo -e "SSH tunnel:     ${GREEN}$SSH_TARGET:$REMOTE_PORT -> localhost:$SSH_TUNNEL_PORT${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for all processes
wait

