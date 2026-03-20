# ============================================================================
# OrthoSSM / Chimera + SpectralVSA — RunPod Private Template Image
# ============================================================================
#
# Base: pytorch/pytorch official image (CUDA 12.4 + cuDNN 9 + PyTorch 2.5.1)
# Python: 3.12  |  CUDA: 12.4  |  cuDNN: 9
#
# What's inside:
#   • PyTorch 2.5.1+cu124 (BF16, TF32, CUDA Graphs)
#   • Mamba-SSM 2.3.0    (SSM scan + Mamba2)
#   • Causal-Conv1D 1.6.0
#   • Triton 3.1.0       (custom kernels: SLR diff-attn, token-err)
#   • Einops, Transformers, Datasets, Tokenizers
#   • JupyterLab (puerto 8888) + SSH (puerto 22) para RunPod
#   • Full Chimera+SpectralVSA codebase en /workspace
#
# Build & Push (Docker Hub privado):
#   docker build -t TUUSUARIO/orthossm:latest .
#   docker login
#   docker push TUUSUARIO/orthossm:latest
#
# RunPod Template config:
#   Image:        TUUSUARIO/orthossm:latest
#   Container disk: 20 GB
#   Expose ports: 8888 (Jupyter), 22 (SSH)
#   Start command: /start.sh
# ============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="OrthoSSM"
LABEL description="Chimera + SpectralVSA v2 — RunPod private template"
LABEL cuda="12.4"
LABEL pytorch="2.5.1"
LABEL python="3.12"

# ── System deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        vim \
        htop \
        nvtop \
        build-essential \
        ninja-build \
        libopenmpi-dev \
        # SSH: requerido por RunPod para acceso remoto al pod
        openssh-server \
        # JupyterLab: UI web en puerto 8888
        jupyter \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Environment variables ─────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES: controlled at runtime
ENV PYTHONUNBUFFERED=1
# Permite TF32 en matmul (≈2× faster en H100/H200/B200, sin pérdida visible)
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# cuBLAS workspace: 256 MiB por device (reduce overhead de selección de alg.)
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# Triton kernel cache persistente en /tmp (survives across containers if /tmp mounted)
ENV TRITON_CACHE_DIR=/tmp/triton_cache
# Evitar warnings de fragmentación de kernels Triton en multi-GPU
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# ── Python deps — tier 1: base científico ────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        # Triton (required by Mamba2 + SLR custom kernels)
        triton==3.1.0 \
        # Einops (tensor rearrange usado en attn)
        einops==0.8.2 \
        # Transformers / tokenizers (tokenizer, HF API)
        transformers>=4.45.0 \
        tokenizers>=0.20.0 \
        datasets>=3.0.0 \
        accelerate>=0.34.0 \
        # Utilidades de training
        tqdm \
        wandb \
        tensorboard \
        # Numerical / analysis
        numpy \
        scipy \
        matplotlib \
        # JupyterLab (RunPod UI)
        jupyterlab==4.1.8 \
        ipywidgets

# ── Python deps — tier 2: Mamba-SSM + Causal-Conv1D ─────────────────────────
# Instalamos desde PyPI directamente con CUDA precompilado.
# Los wheels de PyPI de mamba-ssm 2.3.0 incluyen cu12 para Python 3.12.
# Si la red no está disponible durante el build, sustituir con COPY + pip install
# usando los .whl precompilados del repo:
#   COPY mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl /tmp/
#   COPY causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl /tmp/
#   RUN pip install /tmp/mamba_ssm-2.3.0*.whl /tmp/causal_conv1d-1.6.0*.whl
RUN pip install --no-cache-dir \
        causal-conv1d==1.6.0 \
        mamba-ssm==2.3.0

# ── Workspace setup ───────────────────────────────────────────────────────────
WORKDIR /workspace

# Copiar el codebase completo. .dockerignore excluirá venv/, __pycache__, *.whl.
COPY . /workspace/

# Asegurar que chimera_h200 esté en el PYTHONPATH para imports relativos
ENV PYTHONPATH=/workspace/chimera_h200:/workspace${PYTHONPATH:+:${PYTHONPATH}}

# ── SSH setup (RunPod) ────────────────────────────────────────────────────────
# RunPod inyecta la clave pública del usuario via variable RUNPOD_PUBLIC_KEY.
# El script /start.sh la añade a authorized_keys antes de arrancar sshd.
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
    mkdir -p /var/run/sshd && \
    # PermitRootLogin requerido: RunPod conecta como root
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    # Aumentar keepalive para pods long-running
    echo 'ClientAliveInterval 60' >> /etc/ssh/sshd_config && \
    echo 'ClientAliveCountMax 3'  >> /etc/ssh/sshd_config

# ── Start script ─────────────────────────────────────────────────────────────
# RunPod llama a CMD al arrancar el pod. Este script:
#   1. Instala la clave pública SSH del usuario (via RUNPOD_PUBLIC_KEY)
#   2. Arranca sshd en background
#   3. Arranca JupyterLab en puerto 8888 en background
#   4. Queda en sleep ∞ (el pod no muere cuando termina el CMD)
RUN printf '#!/bin/bash\n\
set -e\n\
# 1. SSH key de RunPod\n\
if [ -n "$RUNPOD_PUBLIC_KEY" ]; then\n\
    echo "$RUNPOD_PUBLIC_KEY" >> /root/.ssh/authorized_keys\n\
    chmod 600 /root/.ssh/authorized_keys\n\
fi\n\
# 2. sshd\n\
service ssh start\n\
# 3. JupyterLab (sin token para acceso directo via RunPod proxy)\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \\\n\
    --NotebookApp.token="" --NotebookApp.password="" \\\n\
    --notebook-dir=/workspace &\n\
# 4. Keep alive\n\
sleep infinity\n' > /start.sh && chmod +x /start.sh

EXPOSE 22 8888

# ── Smoke test: verificar imports críticos ────────────────────────────────────
# Falla el build si algo no está instalado correctamente.
RUN python3 -c "\
import torch; \
import triton; \
from mamba_ssm import Mamba2; \
import causal_conv1d; \
print(f'torch={torch.__version__}  CUDA={torch.version.cuda}  triton={triton.__version__}'); \
print(f'Mamba2 import OK  |  causal_conv1d import OK'); \
assert torch.cuda.is_available() or True  # GPU only available at runtime \
"

# ── Default entrypoint ────────────────────────────────────────────────────────
# RunPod ejecuta este CMD al desplegar el pod.
# Arranca SSH + JupyterLab en background y mantiene el pod vivo.
CMD ["/start.sh"]
