import re

with open("chimera_experiment/advanced_chimera.py", "r") as f:
    text = f.read()

# Extract the AsyncLightBus class
match = re.search(r'(class AsyncLightBus.*?)(?=EOF|$)', text, re.DOTALL)
if match:
    bus_code = match.group(1)
    
    # Remove it from the end of the file
    text = text.replace(bus_code, "")
    
    # Insert before AdvancedChimeraLayer
    idx = text.find("class AdvancedChimeraLayer(nn.Module):")
    if idx != -1:
        text = text[:idx] + bus_code + "\n\n" + text[idx:]

# Fix the duplicate imports at the end
text = text.replace("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\n", "")

# Rewrite the forward pass of AdvancedChimeraLayer to use bus
def update_forward(content):
    sig = "def forward(self, x):"
    new_sig = "def forward(self, x, bus_cache=None):"
    content = content.replace(sig, new_sig)
    
    ret_old = "return x + out"
    ret_new = "out, new_cache = self.bus(out, bus_cache)\n        return x + out, new_cache"
    content = content.replace(ret_old, ret_new)
    
    return content

text = update_forward(text)

with open("chimera_experiment/advanced_chimera.py", "w") as f:
    f.write(text)
