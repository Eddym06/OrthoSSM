import re

with open('/home/OrthoSSM/chimera_h200/spectral_vsa_archive_v2.py', 'r') as f:
    text = f.read()

# 1. OPT-1: Fix SpectralVSA condition number
# Avoid exploding condition numbers by clamping the min denominator,
# and implicitly bounding kappa.
old_kappa = "kappa = norms_positive.max() / norms_positive.min()"
new_kappa = "kappa = norms_positive.max() / norms_positive.clamp(min=1e-3).min() # OPT-1: Ridge-like condition clamp to avoid exploding condition number from vanishing bands"

text = text.replace(old_kappa, new_kappa)

if "OPT-1" not in text:
    print("Could not find condition number update!")

with open('/home/OrthoSSM/chimera_h200/spectral_vsa_archive_v2.py', 'w') as f:
    f.write(text)
print("Updated VSA")
