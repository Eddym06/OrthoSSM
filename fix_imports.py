with open("chimera_experiment/advanced_chimera.py", "r") as f:
    text = f.read()

text = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\n" + text

with open("chimera_experiment/advanced_chimera.py", "w") as f:
    f.write(text)
