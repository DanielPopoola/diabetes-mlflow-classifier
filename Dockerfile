FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI="http://0.0.0.0:5000"


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p mlruns_server/artifacts \
    && mkdir -p src \
    && mkdir -p dataset \
    && mkdir -p notebooks

COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY dataset/ ./dataset/

COPY *.py ./
COPY *.ipynb ./

RUN useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow /app


USER mlflow

EXPOSE 5000

EXPOSE 8888

RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start MLflow server in background\n\
mlflow server \\\n\
  --backend-store-uri sqlite:///mlruns_server/mlflow.db \\\n\
  --default-artifact-root ./mlruns_server/artifacts \\\n\
  --host 0.0.0.0 \\\n\
  --port 5000 &\n\
\n\
# Wait for MLflow server to start\n\
echo "Waiting for MLflow server to start..."\n\
sleep 10\n\
\n\
# Check if we should run Jupyter or just keep MLflow running\n\
if [ "$1" = "jupyter" ]; then\n\
    echo "Starting Jupyter notebook..."\n\
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
elif [ "$1" = "train" ]; then\n\
    echo "Running training pipeline..."\n\
    python -c "from notebooks.main import run_training; run_training()"\n\
else\n\
    echo "MLflow server started at http://0.0.0.0:5000"\n\
    echo "Use docker exec to run training or access the container"\n\
    # Keep container running\n\
    tail -f /dev/null\n\
fi' > /app/start.sh

# Make startup script executable
RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["/app/start.sh"]
