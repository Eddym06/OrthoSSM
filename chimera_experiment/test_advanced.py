import torch
from advanced_chimera import AdvancedChimeraLayer
import time

print("Initializing ADVANCED Mamba 2 CHIMERA with Bus dim 128...")

# Create multiple layers to test the bus passing
layer1 = AdvancedChimeraLayer(d_model=256, expand=2, headdim=32).cuda()
layer2 = AdvancedChimeraLayer(d_model=256, expand=2, headdim=32).cuda()
layer3 = AdvancedChimeraLayer(d_model=256, expand=2, headdim=32).cuda()

layer1.train()
layer2.train()
layer3.train()

x = torch.randn(2, 512, 256, device="cuda", requires_grad=True)

t0 = time.time()

# Pass through 3 layers, acumulando bus cache inter-capa
print("Passing through Layer 1...")
out1, bus_cache = layer1(x, bus_cache=None)
print(f"  Bus cache shape after L1: {bus_cache.shape}")

print("Passing through Layer 2...")
out2, bus_cache = layer2(out1, bus_cache=bus_cache)
print(f"  Bus cache shape after L2: {bus_cache.shape}")

print("Passing through Layer 3...")
out3, bus_cache = layer3(out2, bus_cache=bus_cache)
print(f"  Bus cache shape after L3: {bus_cache.shape}")

loss = out3.sum()
print("Computing backward pass...")
loss.backward()
torch.cuda.synchronize()

t1 = time.time()

print(f"\nForward + Backward multi-layer OK — Shape: {out3.shape}")
print(f"Time: {(t1 - t0)*1000:.2f} ms")

norm_out = layer1.norm(x.detach())
tier_probs = layer1.router(norm_out)
print(f"Layer 1 Router (FAST/HYBRID/FULL): {[f'{p:.3f}' for p in tier_probs[0].tolist()]}")
slr = layer1.slr
print(f"SGR top-K: {int(slr.sgr.top_k_frac*512)}/512 tokens ({slr.sgr.top_k_frac*100:.1f}%)")
print(f"λ diferencial: {slr.lam_logit.sigmoid().item():.4f}")
print(f"Bus dim: {layer1.bus.bus_dim}")
print("\nSUCCESS: 3-layer CHIMERA con SGR+SLR(Triton)+Bus(128d) funcionando.")

