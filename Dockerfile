FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
LABEL org.opencontainers.image.description="Prediction runner" \
      org.opencontainers.image.source="https://github.com/th86/cm4ai-runner"
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash tar ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126 \
    && pip install numpy==2.5.4 pillow==13.0.0 \
    && python3 -c "import torch; assert torch.__version__ == '2.14.1+cu126'"
COPY model.py predict.py /opt/runner/
RUN python3 -m py_compile /opt/runner/model.py /opt/runner/predict.py \
    && python3 /opt/runner/predict.py --help
