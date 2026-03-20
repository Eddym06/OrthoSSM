#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# CHIMERA H200 DEPLOYMENT SCRIPT
# ══════════════════════════════════════════════════════════════════════════════
# Uso:
#   SSH_HOST=root@IP SSH_PORT=PORT ./deploy_h200.sh
#
# Este script:
#   1. Sube todos los archivos Python del proyecto
#   2. Sube los wheels de mamba_ssm y causal_conv1d
#   3. Instala dependencias en el pod
#   4. Descarga el dataset Marin (si no existe)
#   5. Lanza el entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuración ────────────────────────────────────────────────────────────
SSH_HOST="${SSH_HOST:-root@38.80.152.147}"
SSH_PORT="${SSH_PORT:-30445}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

REMOTE_DIR="/root/chimera_h200"
REMOTE_DATA="/root/marin_tokens"
REMOTE_CKPT="/root/ckpt_chimera"

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEEL_DIR="/home/OrthoSSM"

# Archivos esenciales para training (sin tests ni scripts legacy)
ESSENTIAL_FILES=(
    advanced_chimera.py
    chimera_config.py
    chimera_lm.py
    chimera_losses.py
    download_marin_v2.py
    gpu_profile.py
    inference_test.py
    landmark_native.py
    sgr_slr.py
    tokenize_dataset.py
    train_h200_elite.py
    ttt_kernel.py
    ttt_utils.py
)

ssh_cmd() { ssh -p "$SSH_PORT" -i "$SSH_KEY" $SSH_OPTS "$SSH_HOST" "$@"; }
scp_cmd() { scp -P "$SSH_PORT" -i "$SSH_KEY" $SSH_OPTS "$@"; }

echo "══════════════════════════════════════════════════════════════"
echo "  CHIMERA H200 DEPLOYMENT"
echo "  Target: $SSH_HOST:$SSH_PORT"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Test conexión ─────────────────────────────────────────────────────────
echo "[1/6] Testing connection..."
ssh_cmd "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

# ── 2. Crear directorios ─────────────────────────────────────────────────────
echo "[2/6] Creating directories..."
ssh_cmd "mkdir -p $REMOTE_DIR $REMOTE_DATA $REMOTE_CKPT"

# ── 3. Upload archivos Python ────────────────────────────────────────────────
echo "[3/6] Uploading Python files..."
for f in "${ESSENTIAL_FILES[@]}"; do
    echo "  → $f"
    scp_cmd "$LOCAL_DIR/$f" "$SSH_HOST:$REMOTE_DIR/$f"
done

# ── 4. Upload e instalar wheels ──────────────────────────────────────────────
echo "[4/6] Uploading and installing wheels..."

# Detectar si el pod tiene torch 2.6 o 2.5
TORCH_VER=$(ssh_cmd "python3 -c 'import torch; print(torch.__version__[:3])'")
echo "  PyTorch version: $TORCH_VER"

if [[ "$TORCH_VER" == "2.6" ]]; then
    MAMBA_WHEEL="mamba_ssm-2.3.0+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    CC1D_WHEEL="causal_conv1d-1.6.0+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
elif [[ "$TORCH_VER" == "2.5" ]]; then
    MAMBA_WHEEL="mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl"
    CC1D_WHEEL="causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl"
else
    echo "  WARNING: Torch $TORCH_VER — using default wheels, may need compilation"
    MAMBA_WHEEL="mamba_ssm-2.3.0+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    CC1D_WHEEL="causal_conv1d-1.6.0+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
fi

# Check si ya están instalados
NEED_WHEELS=$(ssh_cmd "python3 -c 'import mamba_ssm; import causal_conv1d; print(\"ok\")' 2>/dev/null || echo 'need'")

if [[ "$NEED_WHEELS" == "need" ]]; then
    echo "  Uploading wheels..."
    scp_cmd "$WHEEL_DIR/$MAMBA_WHEEL" "$SSH_HOST:/tmp/$MAMBA_WHEEL"
    scp_cmd "$WHEEL_DIR/$CC1D_WHEEL" "$SSH_HOST:/tmp/$CC1D_WHEEL"
    echo "  Installing..."
    ssh_cmd "pip install --no-deps --break-system-packages /tmp/$CC1D_WHEEL /tmp/$MAMBA_WHEEL && rm /tmp/*.whl"
else
    echo "  mamba_ssm + causal_conv1d already installed ✓"
fi

# Instalar dependencias extra
ssh_cmd "pip install --break-system-packages transformers datasets huggingface_hub zarr 2>/dev/null | tail -1"

# ── 5. Dataset ───────────────────────────────────────────────────────────────
echo "[5/6] Checking dataset..."
HAS_DATA=$(ssh_cmd "test -f $REMOTE_DATA/meta.json && echo yes || echo no")

if [[ "$HAS_DATA" == "no" ]]; then
    echo "  Dataset not found — downloading Marin tokens..."
    echo "  This will take ~10 minutes..."
    ssh_cmd "cd $REMOTE_DIR && python3 download_marin_v2.py --out_dir $REMOTE_DATA --max_tokens 2.6e9"
else
    echo "  Dataset found ✓"
    ssh_cmd "python3 -c \"import json; m=json.load(open('$REMOTE_DATA/meta.json')); print(f'  Shards: {len(m[\"shards\"])}  Tokens: {m[\"total_tokens\"]/1e9:.2f}B  Vocab: {m[\"vocab_size\"]}')\""
fi

# ── 6. Launch training ──────────────────────────────────────────────────────
echo "[6/6] Launching training..."
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Ready to launch. Command:"
echo ""
echo "  nohup python3 $REMOTE_DIR/train_h200_elite.py \\"
echo "      --data_dir $REMOTE_DATA \\"
echo "      --model 125M --vocab 128002 --batch 64 \\"
echo "      --compile --total_tokens 2.6e9 \\"
echo "      --ckpt_dir $REMOTE_CKPT --log_every 10 \\"
echo "      > /root/train.log 2>&1 &"
echo ""
echo "  Monitor: ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'tail -f /root/train.log'"
echo "═══════════════════════════════════════════════════════════"

read -p "Launch now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh_cmd "nohup python3 $REMOTE_DIR/train_h200_elite.py \
        --data_dir $REMOTE_DATA \
        --model 125M --vocab 128002 --batch 64 \
        --compile --total_tokens 2.6e9 \
        --ckpt_dir $REMOTE_CKPT --log_every 10 \
        > /root/train.log 2>&1 &"
    echo "Training launched! PID saved."
    echo "Monitor: ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'tail -f /root/train.log'"
else
    echo "Aborted. Run the command manually when ready."
fi
