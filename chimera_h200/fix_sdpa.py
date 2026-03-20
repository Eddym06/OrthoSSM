import re

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'r') as f:
    text = f.read()

# Remove the broken MultiheadAttention
text = re.sub(
    r"self\.hybrid_attn = nn\.MultiheadAttention\(.*?\)",
    """# SDPA-based O(N) Hybrid Attention
            self.num_heads = d_model // headdim
            self.headdim = headdim
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)""",
    text
)

# Now inject the forward logic for SDPA right after Mamba block or at the end of the layer forward.
# Let's find a good spot in the forward pass.
# Maybe we can inject it right after the mamba_out computation, or on the main residual `x_res`.
# Looking for `mamba_out = self.mamba2(x_norm)`

replacement_forward = """
        # === Mamba2 gradient stabilization =====================================
"""

sdpa_forward = """
        # --- Local/Global Hybrid Attention (SDPA, O(N) Memory) ---
        if getattr(self, 'use_hybrid_attn', False):
            # Attention parallel to the main path, using residual x
            x_attn_norm = self.hybrid_norm(x)
            
            # Projections
            q = self.q_proj(x_attn_norm)
            k = self.k_proj(x_attn_norm)
            v = self.v_proj(x_attn_norm)
            
            B_sz, S_len, _ = q.shape
            
            q = q.view(B_sz, S_len, self.num_heads, self.headdim).transpose(1, 2)
            k = k.view(B_sz, S_len, self.num_heads, self.headdim).transpose(1, 2)
            v = v.view(B_sz, S_len, self.num_heads, self.headdim).transpose(1, 2)
            
            # FlashAttention-2 / SDPA backend
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            attn_out = attn_out.transpose(1, 2).contiguous().view(B_sz, S_len, self.d_model)
            x = x + self.o_proj(attn_out)

        # === Mamba2 gradient stabilization =====================================
"""

text = text.replace(replacement_forward, sdpa_forward)

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'w') as f:
    f.write(text)

print("Injected SDPA forward")
