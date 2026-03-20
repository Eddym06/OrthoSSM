import torch
import torch.nn.functional as F

def ttt_lion_step(dt_bias_adapt, dt_momentum, grad, lr=1e-3, beta=0.9):
    """
    Lion optimizer step for TTT-lite updates on dt_bias.
    dt_bias_adapt: [B, nH]
    dt_momentum: [B, nH]
    grad: [B, nH]
    """
    # Lion update
    c = beta * dt_momentum + (1 - beta) * grad
    update = torch.sign(c)
    dt_bias_adapt.data.sub_(lr * update)

    # Momentum update
    dt_momentum.data.mul_(beta).add_((1 - beta) * grad)
    return dt_bias_adapt, dt_momentum

def ttt_loss_proxy(chunk_out):
    """
    Simulated proxy task for TTT: minimize local variance or self-reconstruction.
    Here we just use variance reduction for simplicity.
    chunk_out: [B, L, nH, hD]
    """
    return chunk_out.var(dim=(1, 3)).mean()

