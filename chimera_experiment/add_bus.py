import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AsyncLightBus(nn.Module):
    """
    AsyncLightBus from OrthoSSM plan.
    Provides a fast cross-layer communication channel.
    """
    def __init__(self, d_model: int, bus_dim: int = 64):
        super().__init__()
        self.bus_dim = bus_dim
        self.publish = nn.Linear(d_model, bus_dim, bias=False)
        self.gather_q = nn.Linear(d_model, bus_dim, bias=False)
        self.modulate = nn.Linear(bus_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor, bus_cache: torch.Tensor = None):
        """
        x: [B, S, D]
        bus_cache: [B, num_layers_so_far, bus_dim] (Optional)
        Returns:
            x_out: modulated x [B, S, D]
            new_cache: updated bus cache [B, num_layers_so_far + 1, bus_dim]
        """
        B, S, D = x.shape
        
        # 1. Publish summary of current layer
        summary = self.publish(x.mean(dim=1)) # [B, bus_dim]
        summary = F.normalize(summary, p=2, dim=-1)
        summary_unsqueezed = summary.unsqueeze(1) # [B, 1, bus_dim]
        
        # 2. Gather from previous layers if cache exists
        if bus_cache is None or bus_cache.shape[1] == 0:
            return x, summary_unsqueezed
            
        q = self.gather_q(x) # [B, S, bus_dim]
        
        # attention: [B, S, bus_dim] x [B, bus_dim, L] -> [B, S, L]
        scores = torch.bmm(q, bus_cache.transpose(1, 2)) / math.sqrt(self.bus_dim)
        attn = F.softmax(scores, dim=-1)
        
        # gathered: [B, S, L] x [B, L, bus_dim] -> [B, S, bus_dim]
        gathered = torch.bmm(attn, bus_cache)
        
        # 3. Modulate
        modulation = self.modulate(gathered) * torch.sigmoid(self.gate)
        
        # 4. Update cache
        new_cache = torch.cat([bus_cache, summary_unsqueezed], dim=1)
        
        return x + modulation, new_cache

