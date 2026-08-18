# Multi-arch base: resolves to arm64 on Apple Silicon, amd64 on Intel.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No dataset COPY and no curl: load_breast_cancer() ships inside the
# scikit-learn wheel, so the image is fully self-contained and offline.

# Optional: lets you exec into the container and run the function by hand.
# Katib overrides the container command anyway (bash -c "<serialised function>"),
# so no ENTRYPOINT is required for the tune() API.
COPY train_func.py /app/train_func.py
