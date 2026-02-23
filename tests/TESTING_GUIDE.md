# Testing & Deployment Guide

## Virtual Environment Setup

To run this project in isolation:

1.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    ```

2.  **Activate the environment**:
    *   MacOS/Linux: `source venv/bin/activate`
    *   Windows: `venv\Scripts\activate`

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode**:
    ```bash
    pip install -e .
    ```

## Running Tests

All tests are written using `pytest`.

1.  **Run all tests**:
    ```bash
    pytest
    ```

2.  **Run with coverage**:
    ```bash
    pip install pytest-cov
    pytest --cov=src tests/
    ```

## Docker Deployment

1.  **Build the image**:
    ```bash
    docker build -t sec-alpha-sentinel:latest .
    ```

2.  **Run the container**:
    ```bash
    docker run -p 8000:8000 \
      -e ANTHROPIC_API_KEY=your_key \
      -e QDRANT_HOST=host.docker.internal \
      sec-alpha-sentinel:latest
    ```
    *Note: `host.docker.internal` allows the container to access Qdrant running on your host machine.*

## Kubernetes Deployment

1.  **Create Secrets**:
    ```bash
    kubectl create secret generic sec-alpha-secrets \
      --from-literal=anthropic-api-key=your_actual_key
    ```

2.  **Deploy**:
    ```bash
    kubectl apply -f k8s-deployment.yaml
    ```
