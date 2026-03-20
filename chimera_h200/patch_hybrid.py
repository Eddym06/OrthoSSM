import re

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'r') as f:
    text = f.read()

# Add a hybrid self-attention module
# It should be added in __init__ for certain layers
init_code = """
        if self.mamba2.A_log.shape == (n_heads,):
            self.mamba2.A_log = nn.Parameter(A_log_tensor)
        else:
            self.mamba2.A_log = nn.Parameter(A_log_tensor.view(-1, 1))
            
        # OPT-3: Atención Híbrida (Sliding Window / Global)
        # Se añade en capas específicas (ej. cada 4 capas) para resolver el "needle in a haystack" >32K
        self.use_hybrid_attn = (layer_idx % 4 == 3)
        if self.use_hybrid_attn:
            self.hybrid_attn = nn.MultiheadAttention(d_model, num_heads=d_model // headdim, batch_first=True)
            self.hybrid_norm = nn.RMSNorm(d_model)
"""

text = re.sub(
    r"(\s+if self\.mamba2\.A_log\.shape .*?self\.mamba2\.A_log = nn\.Parameter\(A_log_tensor\.view\(-1, 1\)\))",
    init_code,
    text,
    flags=re.DOTALL
)

forward_code = """
        # --- Local/Global Hybrid Attention (OPT-3) ---
        if getattr(self, 'use_hybrid_attn', False):
            # Apply attention in parallel or sequentially. Minimal overhead: Additive parallel.
            x_attn_norm = self.hybrid_norm(x)
            attn_out, _ = self.hybrid_attn(x_attn_norm, x_attn_norm, x_attn_norm, need_weights=False)
            x = x + attn_out
"""

text = re.sub(
    r"(\s+# --- 3\. Opcional: MoE/CAS Dispatch ---)",
    forward_code + r"\n\1",
    text
)

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'w') as f:
    f.write(text)
print("Updated Hybrid Attention")
