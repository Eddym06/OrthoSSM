import torch
import numpy as np
import os

token_dtype = np.uint16
# Create dummy file
data = np.arange(100000, dtype=token_dtype)
data.tofile("dummy.bin")

mmap_data = np.memmap("dummy.bin", dtype=token_dtype, mode='r')

ptr = 0
B, T = 4, 16 

# Sequential read using view
batch = torch.from_numpy(mmap_data[ptr : ptr + B*T].astype(np.int64)).view(B, T)
print(batch.shape)
