
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.suppress_errors = True








from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, L_x_ : torch.Tensor):
        l_x_ = L_x_
        reshape = l_x_.reshape([2, 256, 16, 32]);  l_x_ = None
        return (reshape,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('bcf1d2b9059964f25bfbd69aaa6318d2fa8290ca', 2654208, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2, 256, 512), (331776, 1296, 1), requires_grad=True, is_leaf=True)  # L_x_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='minify',
        save_dir='/home/OrthoSSM/chimera_experiment/torch_compile_debug/run_2026_03_08_14_19_28_256265-pid_19406/minifier/checkpoints', autocast=False, backend='inductor')
