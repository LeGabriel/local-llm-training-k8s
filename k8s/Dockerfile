FROM python:3.11-slim

# System deps: build-essential for native extensions (torch, etc.),
# curl + jq for K8s API queries in the entrypoint script (step 1.1.5).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the files needed to install the package.
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the package with the MLflow extra (available for optional tracking).
RUN pip install --no-cache-dir '.[mlflow]'

# Copy the K8s entrypoint script (DDP env var bootstrap for IndexedJob).
COPY k8s/entrypoint.sh /app/k8s/entrypoint.sh
RUN chmod +x /app/k8s/entrypoint.sh

ENTRYPOINT ["python", "-m", "llmtrain"]
CMD ["--help"]
