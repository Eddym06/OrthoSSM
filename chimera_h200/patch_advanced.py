import re

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'r') as f:
    text = f.read()

# 1. Update Mamba2 kwargs
text = text.replace(
    "layer_idx=layer_idx,",
    "layer_idx=layer_idx,\n            use_mem_eff_path=False, # OPT-2: Evitar graph breaks del custom op en fwd"
)

# Wait, if use_triton is the name, I'll pass both if possible, but python **kwargs might crash if it doesn't exist.
# Let's check if 'use_mem_eff_path=False' prevents graph breaks. Yes, it forces fallback to mamba_chunk_scan_combined which is Triton!

# 2. Add selective sublayer checkpointing logic properly
# The prompt says: "La implementación actual ckptea el layer entero pero no las activaciones internas del Mamba2 scan."
# Mamba2's inner activations usually can be checkpointed by checkpointing `mamba_chunk_scan_combined`. But wait, in advanced_chimera.py, it already DOES `_ckpt(self.mamba2, x_norm)`. This checkpoints the whole self.mamba2. Does Mamba2 have its own inner scan? Yes. Checkpointing self.mamba2 does exactly checkpoint the inner activations of Mamba2. Still, maybe it says "ckptea el layer entero pero no las activaciones internas". Maybe the Mamba2 object has a huge output?
# Let's fix 2 first. 

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'w') as f:
    f.write(text)
print("Updated Mamba2")
