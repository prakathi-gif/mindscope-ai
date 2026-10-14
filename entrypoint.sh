#!/bin/bash
set -e

echo "=========================================================="
echo "Starting MindScope AI Container Services"
echo "=========================================================="

MODE=${DEPLOY_MODE:-"all"}

if [ "$MODE" = "api" ]; then
    echo "Starting FastAPI Inference Service on port 8000..."
    exec uvicorn api:app --host 0.0.0.0 --port 8000
elif [ "$MODE" = "ui" ]; then
    echo "Starting Streamlit Dashboard on port 8501..."
    exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
else
    echo "Starting FastAPI REST API in background (port 8000)..."
    uvicorn api:app --host 0.0.0.0 --port 8000 &
    API_PID=$!

    echo "Starting Streamlit Dashboard in foreground (port 8501)..."
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
    ST_PID=$!

    # Trap termination signals and kill both child processes
    trap "echo 'Shutting down services...'; kill -TERM $API_PID $ST_PID; exit 0" SIGINT SIGTERM

    wait
fi

