with open("chimera_experiment/advanced_chimera.py", "r") as f:
    text = f.read()

text = text.replace("def __init__(self, d_model: int, bus_dim: int = 64):", "def __init__(self, d_model: int, bus_dim: int = 128):")

with open("chimera_experiment/advanced_chimera.py", "w") as f:
    f.write(text)
