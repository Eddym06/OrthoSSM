import re

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'r') as f:
    text = f.read()

# Fix the use_mem_eff_path=False mess
text = text.replace("layer_idx=layer_idx,\n            use_mem_eff_path=False, # OPT-2: Evitar graph breaks del custom op en fwd", "layer_idx=layer_idx,")

with open('/home/OrthoSSM/chimera_h200/advanced_chimera.py', 'w') as f:
    f.write(text)

print("Fixed use_mem_eff_path")
