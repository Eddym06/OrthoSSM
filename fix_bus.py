import re

with open("chimera_experiment/advanced_chimera.py", "r") as f:
    content = f.read()

# Separate the bus definition that we just appended to the end
parts = content.split("class AsyncLightBus(nn.Module):")
if len(parts) > 2:
    base_content = parts[0]
    bus_content = "class AsyncLightBus(nn.Module):" + parts[-1]
    
    # Insert bus_content before AdvancedChimeraLayer
    layer_idx = base_content.find("class AdvancedChimeraLayer(nn.Module):")
    final_content = base_content[:layer_idx] + bus_content + "\n\n" + base_content[layer_idx:]
    
    with open("chimera_experiment/advanced_chimera.py", "w") as f:
        f.write(final_content)
