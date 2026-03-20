with open("chimera_experiment/advanced_chimera.py") as f:
    text = f.read()

text = text.replace("self.mamba2.dt_bias = dt_bias_adapt", "self.mamba2.__dict__['dt_bias'] = dt_bias_adapt")
text = text.replace("self.mamba2.dt_bias = orig_dt_bias", "self.mamba2.__dict__['dt_bias'] = orig_dt_bias")

with open("chimera_experiment/advanced_chimera.py", "w") as f:
    f.write(text)
