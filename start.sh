#!/bin/bash

set -e

uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 &

API_PID=$!

streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501

wait $API_PID