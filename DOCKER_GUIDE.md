# OrthoSSM Docker Deploy Guide — RunPod

## 1. Build y push al registry privado (Docker Hub)

```bash
# Reemplaza TUUSUARIO con tu Docker Hub username
export DOCKER_USER=TUUSUARIO

# Build
docker build -t $DOCKER_USER/orthossm:latest .

# Login (pide usuario + password / access token)
docker login

# Push al repo privado
docker push $DOCKER_USER/orthossm:latest
```

> El repo en Docker Hub debe estar creado como **Private** antes del push:
> hub.docker.com → Create Repository → Visibility: Private

---

## 2. Configurar credenciales en RunPod

RunPod → Settings → **Container Credentials** → Add:
- Registry: `docker.io`
- Username: `TUUSUARIO`
- Password: tu Docker Hub password o access token

Esto permite a RunPod hacer `docker pull` del repo privado al desplegar pods.

---

## 3. Crear el Template en RunPod

RunPod → **Manage Templates** → New Template:

| Campo | Valor |
|-------|-------|
| Template name | `OrthoSSM-Chimera` |
| Container image | `TUUSUARIO/orthossm:latest` |
| Container disk | `20 GB` |
| Expose ports | `8888/http, 22/tcp` |
| Start command | `/start.sh` |
| Environment vars | ver tabla abajo |

Variables de entorno recomendadas en el template:

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `PYTHONUNBUFFERED` | `1` | Stdout sin buffer |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `1` | TF32 en H100/H200 |
| `WANDB_API_KEY` | *(tu key)* | Logging de runs |

---

## 4. Desplegar un pod

RunPod → Deploy → selecciona GPU (H100 / H200 / A100 / RTX 4090) → elige tu template `OrthoSSM-Chimera` → Deploy.

Al arrancar el pod:
- **SSH**: `ssh root@<pod-ip> -p <port>` (credencial desde RunPod UI → Connect)
- **JupyterLab**: RunPod → Connect → abre el proxy en puerto 8888

---

## 5. Actualizar la imagen

```bash
# Después de cambios en el codebase
docker build -t $DOCKER_USER/orthossm:latest .
docker push $DOCKER_USER/orthossm:latest
# En RunPod: el próximo pod usa la nueva imagen automáticamente
```

---

## 6. Desarrollo local (sin RunPod)

```bash
# Bash interactivo con GPU local
docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    TUUSUARIO/orthossm:latest /start.sh

# Eval directo sin arrancar Jupyter
docker run --gpus all --rm \
    TUUSUARIO/orthossm:latest \
    python3 /workspace/chimera_h200/eval_arch_300.py
```

---

## 7. Build offline (sin internet, usando wheels locales)

Si el servidor de build no tiene acceso a PyPI:
```dockerfile
# Sustituir en Dockerfile la sección mamba-ssm por:
COPY mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl /tmp/
COPY causal_conv1d-1.6.0-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install /tmp/mamba_ssm-2.3.0*.whl /tmp/causal_conv1d-1.6.0*.whl
```

---

## 8. Verificar GPU dentro del pod

```bash
python3 -c "
import torch
from mamba_ssm import Mamba2
print(torch.cuda.get_device_name(0))
print(f'BF16: {torch.cuda.is_bf16_supported()}  TF32: {torch.backends.cuda.matmul.allow_tf32}')
"
```
