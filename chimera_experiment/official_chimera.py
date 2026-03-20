import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba2

class GatedComplexityPredictor(nn.Module):
    def __init__(self, d_model: int, n_tiers: int = 3):
        super().__init__()
        d_hidden = 32
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_tiers)
        )
    
    def forward(self, x: torch.Tensor):
        x_mean = x.mean(dim=1)  # [B, D]
        logits = self.mlp(x_mean) # [B, n_tiers]
        probs = F.softmax(logits, dim=-1)
        return probs

class OfficialChimeraLayer(nn.Module):
    def __init__(self, d_model: int = 256, expand: int = 2, headdim: int = 32):
        super().__init__()
        self.d_model = d_model
        
        # Base Mamba 2 official
        self.mamba2 = Mamba2(
            d_model=d_model,
            expand=expand,
            headdim=headdim,
            d_state=64
        )
        
        # 1. Multi-scale A init
        # Mamba2 has self.mamba2.A_log of shape [nheads] or [nheads, 1]
        n_heads = self.mamba2.nheads
        lambdas = [math.exp(-0.001 * (2**i)) for i in range(n_heads)]
        A_init = [-math.log(l) for l in lambdas]
        A_log_tensor = torch.tensor([math.log(a) for a in A_init], dtype=torch.float32)
        
        # Override A_log inside Mamba2
        if self.mamba2.A_log.shape == (n_heads,):
            self.mamba2.A_log = nn.Parameter(A_log_tensor)
        else:
            self.mamba2.A_log = nn.Parameter(A_log_tensor.view(-1, 1))
            
        print("Mamba 2: Overridden A_log with OrthoSSM Multi-scale.")

        # 2. Router
        self.router = GatedComplexityPredictor(d_model, 3)
        self.norm = nn.RMSNorm(d_model)

        # 3. TTT state & logic
        self.dt_momentum = None
        self.ttt_lr = 1e-3
        self.ttt_beta = 0.9

    def forward(self, x):
        B, S, D = x.shape
        x_norm = self.norm(x)
        
        # Get tier probabilities
        probs = self.router(x_norm) # [B, 3] -> 0: FAST, 1: HYBRID, 2: FULL
        
        prob_fast = probs[:, 0].view(B, 1, 1)
        prob_hybrid = probs[:, 1].view(B, 1, 1)
        prob_full = probs[:, 2].view(B, 1, 1)
        
        # TTT-lite pseudo-proxy: Predict variance over hidden states as error
        # In a real step, we'd adjust dt_bias before scan. Since Mamba 2 does the scan internally,
        # we compute an error on x_norm to update dt_bias dynamically.
        # This proxy is "variance of x_norm features represents structural complexity".
        if self.dt_momentum is None:
            self.dt_momentum = torch.zeros_like(self.mamba2.dt_bias)

        # TTT on dt_bias if routing indicates active hybrid/full
        # We'll compute a surrogate loss on the layer input itself to shift dt_bias right before processing
        total_active_prob = probs[:, 1:].sum(dim=-1).mean()
        if total_active_prob > 0.3 and self.training:
            # We want dt_bias to adapt before mamba scan.
            # Local proxy loss: maximize information flow (variance) inside current chunk's norm
            local_loss = -x_norm.var(dim=(1,2)).mean()
            
            # Since dt_bias doesn't immediately affect x_norm, we approximate a pseudo-gradient. 
            # In true TTT, we'd do a quick forward pass of a mini-chunk, get loss, update dt_bias.
            # For simplicity: use random noise projected with gradient to simulate adaptation overhead, 
            # or do a 1-step meta-update if we had a backward chunk.
            pseudo_grad = torch.randn_like(self.mamba2.dt_bias) * 0.01
            
            c = self.ttt_beta * self.dt_momentum + (1 - self.ttt_beta) * pseudo_grad
            update = torch.sign(c)
            with torch.no_grad():
                self.mamba2.dt_bias.sub_(self.ttt_lr * update * total_active_prob)
                self.dt_momentum.mul_(self.ttt_beta).add_((1 - self.ttt_beta) * pseudo_grad)

        # Forward via Mamba 2 official
        # In a strict soft-gated setup:
        # We just pass x to Mamba2. FAST = standard, HYBRID/FULL = standard with adaptive dt_bias
        # We blend the output proportionally to show soft-gating logic:
        # Actually Mamba 2 handles the entire transformation. We can simulate a "FAST" skip connection.
        
        mamba_out = self.mamba2(x_norm)
        fast_out = x_norm # Trivial linear or skip path for FAST
        
        # Soft-gating!
        out = prob_fast * fast_out + (prob_hybrid + prob_full) * mamba_out

        return x + out

if __name__ == "__main__":
    print("Initializing Official Mamba 2 CHIMERA...")
    model = OfficialChimeraLayer(d_model=256, expand=2, headdim=32).cuda()
    model.train()
    
    x = torch.randn(2, 512, 256, device="cuda", requires_grad=True)
    out = model(x)
    print("Forward Pass Works! Shape:", out.shape)
