import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedComplexityPredictor(nn.Module):
    def __init__(self, d_model: int, n_tiers: int = 3):
        super().__init__()
        # Tiny router ~8k params -> if d_model=256: 256*32 + 32*3 = 8k
        d_hidden = 32
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, n_tiers)
        )
    
    def forward(self, x: torch.Tensor, training: bool = False):
        # x is [B, S, D]
        x_mean = x.mean(dim=1)  # [B, D]
        logits = self.mlp(x_mean) # [B, n_tiers]
        
        if training:
            # Gumbel Softmax for differentiable routing
            probs = F.gumbel_softmax(logits, tau=1.0, hard=True)
            tier_idx = probs.argmax(dim=-1)
            return tier_idx, probs
        else:
            tier_idx = logits.argmax(dim=-1)
            return tier_idx, F.softmax(logits, dim=-1)

class ChimeraLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, n_heads: int = 8, chunk_size: int = 64):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_model * expand
        self.d_state  = d_state
        self.n_heads  = n_heads
        self.head_dim = self.d_inner // n_heads
        self.chunk_size = chunk_size

        assert self.d_inner % n_heads == 0

        self.in_proj = nn.Linear(
            d_model,
            self.d_inner + self.d_inner + n_heads * d_state * 2 + n_heads,
            bias=False,
        )
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )
        self.dt_proj = nn.Linear(n_heads, self.d_inner, bias=True)

        # Multi-scale A initialization
        # MULTI_SCALE_LAMBDA = [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7] (8 heads)
        # We want A target to be -ln(lambda)
        if n_heads == 8:
            lambdas = [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7]
        else:
            lambdas = [math.exp(-0.001 * (2**i)) for i in range(n_heads)]
            
        A_init = [-math.log(l) for l in lambdas] # Positive values
        # A_log allows us to learn A where A = -exp(A_log). So A_log = log(-(-A_target)) = log(A_init)
        A_log = torch.tensor([math.log(a) for a in A_init], dtype=torch.float32)
        self.A_log = nn.Parameter(A_log)

        self.D = nn.Parameter(torch.ones(n_heads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm_in  = nn.RMSNorm(d_model)
        self.norm_out = nn.RMSNorm(self.d_inner)
        self.router = GatedComplexityPredictor(d_model, 3)

    def _ssd_parallel_scan_ttt(self, x, dt, B_in, C_in, do_ttt):
        B_sz, S, nH, hD = x.shape
        d_st = B_in.shape[-1]
        chunk = self.chunk_size
        
        # Base A
        A = -torch.exp(self.A_log.float())
        
        orig_dtype = x.dtype
        x = x.float()
        dt = dt.float()
        B_in = B_in.float()
        C_in = C_in.float()

        # TTT Momentum initialization (shape corresponding to dt_bias which is scalar per head? Wait, dt is [B, S, nH] originally? No, dt here is [B,S,nH,hD] then reduced. In the mock it was:
        # dt_h = dt.reshape(B_sz, S, self.n_heads, self.head_dim).mean(dim=-1) -> shape [B, S, nH])
        dt_bias_adapt = torch.zeros((B_sz, nH), device=x.device, dtype=torch.float32)
        dt_momentum = torch.zeros((B_sz, nH), device=x.device, dtype=torch.float32)

        h = x.new_zeros(B_sz, nH, hD, d_st)
        outputs = []
        
        ttt_lr = 1e-3
        ttt_beta = 0.9

        for ci in range(math.ceil(S / chunk)):
            t0, t1 = ci * chunk, min((ci + 1) * chunk, S)
            L = t1 - t0

            x_c  = x[:, t0:t1]    # [B, L, nH, hD]
            # Add TTT bias to dt
            if do_ttt:
                dt_c = F.softplus(dt[:, t0:t1] + dt_bias_adapt.unsqueeze(1))
            else:
                dt_c = dt[:, t0:t1]
                
            B_c  = B_in[:, t0:t1] # [B, L, nH, d_st]
            C_c  = C_in[:, t0:t1] # [B, L, nH, d_st]

            # 1. log-cumsum trick
            logA = dt_c * A.unsqueeze(0).unsqueeze(0)  # [B, L, nH]
            logA_cs = logA.cumsum(dim=1)               # [B, L, nH]

            lcs = logA_cs.permute(0, 2, 1)  # [B, nH, L]
            M = (lcs.unsqueeze(3) - lcs.unsqueeze(2)).exp()  # [B, nH, L, L]
            causal_mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
            M = M * causal_mask.unsqueeze(0).unsqueeze(0)

            # 2. Intra-chunk
            BX = B_c.unsqueeze(3) * x_c.unsqueeze(4)  # [B, L, nH, hD, d_st]
            intra_out = torch.zeros(B_sz, L, nH, hD, device=x.device, dtype=x.dtype)
            for hh in range(nH):
                C_h   = C_c[:, :, hh, :]   # [B, L, d_st]
                BX_h  = BX[:, :, hh, :, :] # [B, L, hD, d_st]
                M_h   = M[:, hh, :, :]     # [B, L, L]
                BX_h_T = BX_h.permute(0, 1, 3, 2)  # [B, L_i, d_st, hD]
                BX_flat = BX_h_T.reshape(B_sz, L, d_st * hD)  
                weighted = M_h @ BX_flat   
                weighted = weighted.reshape(B_sz, L, d_st, hD)  
                intra_h = (C_h.unsqueeze(-1) * weighted).sum(dim=2)  
                intra_out[:, :, hh, :] = intra_h

            # 3. Inter-chunk
            decay  = logA_cs.exp()      # [B, L, nH]
            inter_hd = torch.einsum('blhd, bhnd -> blhn', C_c, h)  # [B,L,nH,hD]
            inter = decay.unsqueeze(-1) * inter_hd  # [B,L,nH,hD]

            out_c = intra_out + inter  # [B, L, nH, hD]
            
            # TTT Update based on chunk error (predict next chunk input roughly, or self-reconstruction)
            # In TTT standard we use self-reconstruction or next token prediction. We'll use a simple proxy: variance of output
            if do_ttt and L == chunk:
                # We need a dummy gradient for dt_bias_adapt. 
                # To do TTT in the forward pass implicitly in PyTorch without a full grad graph of the whole model,
                # we can detach, compute local loss, and use autograd.grad on dt_bias_adapt
                # We must ensure dt_bias_adapt has requires_grad
                if dt_bias_adapt.requires_grad == False:
                    dt_bias_adapt.requires_grad_(True)
                loss = out_c.var(dim=(1, 3)).mean()
                grad = torch.autograd.grad(loss, dt_bias_adapt, retain_graph=True, allow_unused=True)[0]
                if grad is not None:
                    # Lion step inline
                    c = ttt_beta * dt_momentum + (1 - ttt_beta) * grad
                    update = torch.sign(c)
                    with torch.no_grad():
                        dt_bias_adapt.sub_(ttt_lr * update)
                        dt_momentum.mul_(ttt_beta).add_((1 - ttt_beta) * grad)


            outputs.append(out_c.detach() if do_ttt else out_c) # keep graph or detach? For test we keep graph out

            # 4. Update SSM state
            total_decay = logA_cs[:, -1, :].exp()              
            h = total_decay.unsqueeze(-1).unsqueeze(-1) * h    
            M_last = M[:, :, -1, :]  
            for hh in range(nH):
                BX_h = BX[:, :, hh, :, :]         
                ml   = M_last[:, hh, :].unsqueeze(-1).unsqueeze(-1)  
                h[:, hh] += (ml * BX_h).sum(dim=1)  

        return torch.cat(outputs, dim=1).to(orig_dtype)

    def forward(self, hidden):
        residual = hidden
        hidden = self.norm_in(hidden)
        B_sz, S, _ = hidden.shape
        
        # Decide tier
        tier_idx, _ = self.router(hidden)
        # tier_idx: [B]
        # For simplicity in batching, we take mode tier or process separately. 
        # If any batch elm is >= 1 (HYBRID/FULL), we do TTT
        tier_max = tier_idx.max().item()
        
        # In a real impl, fast path length check also occurs:
        if S < 384:
            tier_max = 0
            
        do_ttt = (tier_max >= 1)

        proj = self.in_proj(hidden)  
        split_sizes = [self.d_inner, self.d_inner, self.n_heads * self.d_state, self.n_heads * self.d_state, self.n_heads]
        z, x, B_flat, C_flat, dt_raw = proj.split(split_sizes, dim=-1)

        x = self.conv(x.transpose(1, 2))[:, :, :S].transpose(1, 2)
        x = F.silu(x)

        # Softplus is applied IN THE SCAN if do_ttt, so we pass raw
        dt = self.dt_proj(dt_raw)
        dt_h = dt.reshape(B_sz, S, self.n_heads, self.head_dim).mean(dim=-1)
        if not do_ttt:
            dt_h = F.softplus(dt_h)

        B_ssm = B_flat.reshape(B_sz, S, self.n_heads, self.d_state)
        C_ssm = C_flat.reshape(B_sz, S, self.n_heads, self.d_state)
        x_h   = x.reshape(B_sz, S, self.n_heads, self.head_dim)

        y = self._ssd_parallel_scan_ttt(x_h, dt_h, B_ssm, C_ssm, do_ttt)

        y = y + x_h * self.D.view(1, 1, self.n_heads, 1)
        y = y.reshape(B_sz, S, self.d_inner)
        y = y * F.silu(z)
        y = self.norm_out(y)
        y = self.out_proj(y)
        return residual + y
