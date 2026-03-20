import re

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'r') as f:
    text = f.read()

# Fix TTT Gradient stabilization for Hybrid Layers
replacement = """                # Estimador SPSA: g_k = (L⁺ - L⁻) / (2·ε·Δ_k)
                _ttt_grad_to_apply = (loss_p - loss_m) / (2.0 * eps * delta)
                
                # TTT Proxy Sync: En capas de Atención Híbrida, la Mamba2 no tiene toda la carga representacional.
                # Reducimos severamente el gradiente del proxy (TTT blind to attention) para evitar que
                # dt_bias sobrecompense tratando de explicar dinámicas de las que ahora se encarga el transformer.
                if getattr(self, 'use_hybrid_attn', False):
                    _ttt_grad_to_apply = _ttt_grad_to_apply * 0.1
"""

text = text.replace(
    "                # Estimador SPSA: g_k = (L⁺ - L⁻) / (2·ε·Δ_k)\n                _ttt_grad_to_apply = (loss_p - loss_m) / (2.0 * eps * delta)",
    replacement
)

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'w') as f:
    f.write(text)

print("Fixed TTT Proxy")
