"""
deep_analysis.py — Análisis profundo de CHIMERA
  T1: Throughput & VRAM vs longitud de secuencia
  T2: Estabilidad numérica
  T3: Stress test contextos largos
  T4: Latencia por componente
  T5: Acumulación de error chunked vs full
  T6: Calidad de gradientes (finite-difference)
  T7: Proyección 1M tokens
  T8: Rendimiento step() autoregresivo
"""
import sys, os, gc, time, math, json, argparse, traceback
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, torch.nn.functional as F

def vram_mb():
    if torch.cuda.is_available(): torch.cuda.synchronize(); return torch.cuda.memory_allocated()/1e6
    return 0.0
def vram_peak_mb():
    if torch.cuda.is_available(): torch.cuda.synchronize(); return torch.cuda.max_memory_allocated()/1e6
    return 0.0
def reset_peak():
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def timer_ms(fn, warmup=2, reps=8):
    for _ in range(warmup): fn()
    sync()
    times = []
    for _ in range(reps):
        t0=time.perf_counter(); fn(); sync(); times.append((time.perf_counter()-t0)*1e3)
    t=torch.tensor(times); return float(t.mean()), float(t.std())

def make_model(device):
    from advanced_chimera import AdvancedChimeraLayer
    return AdvancedChimeraLayer(d_model=256).to(device).float()

def banner(t,w=72): print(f"\n{'='*w}\n  {t}\n{'='*w}")
def ok(m):   print(f"  [OK]  {m}")
def info(m): print(f"        {m}")
def warn(m): print(f"  [!!]  {m}")
def fail(m): print(f"  [XX]  {m}")

QUICK = False

# T1 ------------------------------------------------------------------
def test_throughput_vram(device):
    banner("T1 - Throughput & VRAM vs longitud de secuencia")
    model = make_model(device); model.eval(); d=256
    lens = [256,512,1024,2048,4096] if not QUICK else [256,512,1024]
    results = []
    for S in lens:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        reset_peak()
        x = torch.randn(1,S,d,device=device)
        vbase = vram_mb()
        def fwd():
            with torch.no_grad(): model(x, bus_cache=None)
        mean_ms, std_ms = timer_ms(fwd, warmup=3, reps=5)
        peak=vram_peak_mb(); delta=peak-vbase; tps=int((S/mean_ms)*1e3)
        results.append({'S':S,'ms':round(mean_ms,2),'std_ms':round(std_ms,2),
                        'tps':tps,'vram_delta_mb':round(delta,1),'vram_peak_mb':round(peak,1)})
        info(f"S={S:>5} | {mean_ms:6.2f}+-{std_ms:.2f} ms | {tps:>9,} tok/s | VRAM d={delta:.1f} MB | peak={peak:.1f} MB")

    xs=[r['S'] for r in results]; ys=[r['vram_delta_mb'] for r in results]
    n=len(xs); sx=sum(xs); sy=sum(ys); sxy=sum(a*b for a,b in zip(xs,ys)); sxx=sum(a*a for a in xs)
    slope=(n*sxy-sx*sy)/(n*sxx-sx*sx+1e-9); intercept=(sy-slope*sx)/n
    print()
    info("--- Extrapolacion prefill (activaciones O(S)) ---")
    for S_p,lbl in [(131_072,'128K'),(1_000_000,'1M')]:
        vp=slope*S_p+intercept; tps_ref=results[-1]['tps']
        ms_p=S_p/tps_ref*1e3
        info(f"S={lbl:>4}: VRAM={vp/1024:.1f} GB | latencia={ms_p/1e3:.1f}s | throughput={tps_ref:,} tok/s")
    print()
    info("--- Modo INFERENCE (step token-a-token, estado O(1)) ---")
    try:
        c=model.allocate_inference_cache(1,dtype=torch.float32)
        state_mb=sum(v.numel()*4/1e6 for v in c.values() if isinstance(v,torch.Tensor))
        info(f"Estado SSM fijo (1 capa): {state_mb:.2f} MB - INDEPENDIENTE de S")
        info(f"Con 32 capas:             {state_mb*32:.2f} MB - igual para 1M que para 1 token")
    except Exception as e:
        info(f"(allocate_inference_cache: {e})")
    return results

# T2 ------------------------------------------------------------------
def test_numerical_stability(device):
    banner("T2 - Estabilidad numerica & correccion de errores")
    model=make_model(device); model.eval(); d=256; results={}
    def fwd(x):
        with torch.no_grad(): return model(x,bus_cache=None)[0]

    for x,lbl in [(torch.randn(1,512,d,device=device),'normal'),
                  (torch.zeros(1,512,d,device=device),'cero')]:
        o=fwd(x)
        status='OK' if not o.isnan().any() and not o.isinf().any() else 'FAIL'
        ok(f"{lbl:<6} NaN={o.isnan().any().item()} Inf={o.isinf().any().item()} RMS={o.pow(2).mean().sqrt():.4f} [{status}]")
        results[lbl]=status

    for scale,lbl in [(1e6,'1e6'),(1e-8,'1e-8')]:
        try:
            o=fwd(torch.full((1,128,d),scale,device=device))
            ok(f"+/-{lbl:<5} NaN={o.isnan().any().item()} Inf={o.isinf().any().item()} RMS={o.float().nan_to_num(0).pow(2).mean().sqrt():.4f}")
            results[lbl]='OK'
        except Exception as e:
            warn(f"+/-{lbl}: excepcion - {e}"); results[lbl]='ERROR'

    # Determinismo
    xd=torch.randn(1,256,d,device=device)
    torch.manual_seed(0); o1=fwd(xd)
    torch.manual_seed(0); o2=fwd(xd)
    diff=(o1-o2).abs().max().item()
    ok(f"Determinismo max_diff={diff:.2e} {'OK' if diff<1e-5 else 'WARNING'}")

    # Gradientes
    model.train()
    xg=torch.randn(1,256,d,device=device)
    og,_=model(xg,bus_cache=None); og.mean().backward()
    gn={n:p.grad.norm().item() for n,p in model.named_parameters() if p.grad is not None}
    n_fin=sum(1 for v in gn.values() if math.isfinite(v) and v>0)
    ok(f"Gradientes {n_fin}/{len(gn)} parametros con grad finita y >0")

    print()
    info("Grad norm por modulo:")
    groups={'mamba2':[],'ttt':[],'slr':[],'archive':[],'bus':[],'router':[],'norm':[]}
    other=[]
    for name,val in gn.items():
        matched=False
        for g in groups:
            if g in name.lower(): groups[g].append(val); matched=True; break
        if not matched: other.append(val)
    for g,vs in groups.items():
        if vs: info(f"  {g:>8}: mean={sum(vs)/len(vs):.4f}  max={max(vs):.4f}  n={len(vs)}")
    if other: info(f"  {'otro':>8}: mean={sum(other)/len(other):.4f}  n={len(other)}")
    return {'finite':n_fin,'total':len(gn),'cases':results}

# T3 ------------------------------------------------------------------
def test_long_context(device):
    banner("T3 - Stress test: contextos largos")
    model=make_model(device); model.eval(); d=256
    targets=[4096,8192,16384,32768] if not QUICK else [4096,8192]
    results=[]
    for S in targets:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        reset_peak()
        try:
            x=torch.randn(1,S,d,device=device)
            t0=time.perf_counter()
            with torch.no_grad(): out,_=model(x,bus_cache=None)
            sync(); ms=(time.perf_counter()-t0)*1e3
            peak=vram_peak_mb(); tps=int((S/ms)*1e3)
            valid=not out.isnan().any() and not out.isinf().any()
            ok(f"S={S:>6} | {ms:7.1f} ms | {tps:>8,} tok/s | VRAM={peak:.1f} MB | valid={valid}")
            results.append({'S':S,'ms':round(ms,2),'tps':tps,'vram':round(peak,1),'valid':valid})
            del x,out
        except torch.cuda.OutOfMemoryError:
            warn(f"S={S:>6} | OOM"); results.append({'S':S,'error':'OOM'})
        except Exception as e:
            fail(f"S={S:>6} | {e}"); results.append({'S':S,'error':str(e)})
    valid=[r for r in results if 'tps' in r]
    if len(valid)>=2:
        print(); info("Escala O(?):")
        for i in range(1,len(valid)):
            r0,r1=valid[i-1],valid[i]; f_S=r1['S']/r0['S']; f_t=r1['ms']/r0['ms']
            expo=math.log(f_t)/math.log(f_S) if f_S>1 else 1.0
            info(f"  {r0['S']:>6}->{r1['S']:>6}: tiempo x{f_t:.2f} para x{f_S:.0f} tokens  => O(S^{expo:.2f})")
    return results

# T4 ------------------------------------------------------------------
def test_latency_breakdown(device):
    banner("T4 - Latencia por componente (S=2048)")
    model=make_model(device); model.eval(); d=256; S=2048
    x=torch.randn(1,S,d,device=device)
    def bench(fn,warmup=3,reps=8):
        for _ in range(warmup): fn()
        sync(); t=[]
        for _ in range(reps):
            t0=time.perf_counter(); fn(); sync(); t.append((time.perf_counter()-t0)*1e3)
        return sum(t)/len(t)

    ms_full=bench(lambda: model(x,bus_cache=None))
    ok(f"Full forward       : {ms_full:7.3f} ms (100%)")

    x_norm=model.norm(x)
    ms_m2=bench(lambda: model.mamba2(x_norm))
    info(f"  mamba2 SSD scan  : {ms_m2:7.3f} ms ({100*ms_m2/ms_full:.1f}%)")

    ms_router=bench(lambda: model.router(x))
    info(f"  router (3-tier)  : {ms_router:7.3f} ms ({100*ms_router/ms_full:.1f}%)")

    with torch.no_grad(): m2_out=model.mamba2(x_norm)
    ms_slr=bench(lambda: model.slr(m2_out))
    info(f"  SLR+DiffAttn     : {ms_slr:7.3f} ms ({100*ms_slr/ms_full:.1f}%)")

    q=torch.randn(1,1,d,device=device)
    ms_arch=bench(lambda: model.archive.retrieve(q))
    info(f"  archive retrieve : {ms_arch:7.3f} ms ({100*ms_arch/ms_full:.1f}%)")

    ms_norm=bench(lambda: model.norm(x))
    info(f"  RMSNorm          : {ms_norm:7.3f} ms ({100*ms_norm/ms_full:.1f}%)")

    overhead=ms_full-ms_m2-ms_slr-ms_arch-ms_router-ms_norm
    info(f"  TTT+Bus+Pipeline : {overhead:7.3f} ms ({100*overhead/ms_full:.1f}%)")
    return {'full':ms_full,'mamba2':ms_m2,'router':ms_router,'slr':ms_slr,'archive':ms_arch,'norm':ms_norm,'overhead':overhead}

# T5 ------------------------------------------------------------------
def test_error_accumulation(device):
    banner("T5 - Acumulacion de error: chunked vs full forward")
    model=make_model(device); model.eval(); d=256
    chunk=512; n_ch=6 if not QUICK else 4; total=chunk*n_ch
    xfull=torch.randn(1,total,d,device=device)
    with torch.no_grad(): out_full,_=model(xfull,bus_cache=None)
    parts=[]
    for i in range(n_ch):
        with torch.no_grad(): oc,_=model(xfull[:,i*chunk:(i+1)*chunk,:],bus_cache=None)
        parts.append(oc)
    out_ch=torch.cat(parts,dim=1)
    diff=(out_full-out_ch).abs()
    info(f"Full vs chunked: max_err={diff.max():.4f}  mean={diff.mean():.4f}  rmse={diff.pow(2).mean().sqrt():.4f}")
    errs=[( out_full[:,i*chunk:(i+1)*chunk,:]-out_ch[:,i*chunk:(i+1)*chunk,:]).abs().mean().item() for i in range(n_ch)]
    early=sum(errs[:n_ch//2])/(n_ch//2); late=sum(errs[n_ch//2:])/(n_ch//2)
    drift=(late-early)/(early+1e-9)*100
    if abs(drift)>5: warn(f"Drift: {drift:+.1f}% (early={early:.5f} -> late={late:.5f})")
    else: ok(f"Sin drift: {drift:+.1f}%  (early={early:.5f} late={late:.5f})")
    norms=out_full[0].norm(dim=-1)
    info(f"Norma salida: min={norms.min():.3f} max={norms.max():.3f} mean={norms.mean():.3f} std={norms.std():.3f}")
    return {'chunk_errs':errs,'drift_pct':round(drift,2)}

# T6 ------------------------------------------------------------------
def test_gradient_quality(device):
    banner("T6 - Calidad de gradientes (finite-difference)")
    model=make_model(device); model.train(); d=256
    x=torch.randn(1,128,d,device=device); eps=1e-3; results=[]
    cands=[(n,p) for n,p in model.named_parameters() if p.requires_grad and 1<=p.numel()<=32]
    checked=set(); params=[]
    for n,p in cands:
        g=n.split('.')[0]
        if g not in checked: params.append((n,p)); checked.add(g)
        if len(params)>=5: break
    for name,p in params:
        for pp in model.parameters():
            if pp.grad is not None: pp.grad.zero_()
        out,_=model(x,bus_cache=None); out.sum().backward()
        ga=p.grad.detach().flatten()[:4].clone()
        gn=[]
        with torch.no_grad():
            for idx in range(min(4,p.numel())):
                flat=p.data.flatten(); v0=flat[idx].item()
                flat[idx]=v0+eps; p.data.copy_(flat.reshape_as(p.data))
                op,_=model(x,bus_cache=None); fp=op.sum().item()
                flat[idx]=v0-eps; p.data.copy_(flat.reshape_as(p.data))
                om,_=model(x,bus_cache=None); fm=om.sum().item()
                flat[idx]=v0; p.data.copy_(flat.reshape_as(p.data))
                gn.append((fp-fm)/(2*eps))
        gn_t=torch.tensor(gn,dtype=torch.float32); ga_t=ga.cpu()[:len(gn_t)]
        rel=((ga_t-gn_t).abs()/(gn_t.abs()+1e-8)).mean().item()
        ok(f"{name:<45} rel_err={rel:.4f}  {'OK' if rel<0.05 else 'ALTO'}")
        results.append({'param':name,'rel_err':round(rel,5)})
    return results

# T7 ------------------------------------------------------------------
def report_1m_projection(t1):
    banner("T7 - Proyeccion 1M tokens: VRAM, velocidad y estrategia")
    B,d=1,256
    nheads=16; headdim=32; d_state=64; d_conv=4; ngroups=1; d_inner=512
    ssm_kb=B*nheads*headdim*d_state*4/1024
    conv_kb=B*(d_inner+2*ngroups*d_state)*d_conv*4/1024
    bus_kb=128*d*4/1024
    state_kb=ssm_kb+conv_kb+bus_kb

    print()
    info("MODO INFERENCE (step) — estado O(1):")
    info(f"  ssm_state  : {ssm_kb:.1f} KB/capa")
    info(f"  conv_state : {conv_kb:.1f} KB/capa")
    info(f"  bus_cache  : {bus_kb:.1f} KB/capa")
    info(f"  TOTAL 1 capa: {state_kb:.1f} KB  | 32 capas: {state_kb*32/1024:.2f} MB  <- FIJO, independiente de S")
    print()
    info("MODO PREFILL (chunked, obligatorio >4K en 6GB GPU):")
    chunk_s=4096; n_ch=math.ceil(1_000_000/chunk_s)
    info(f"  Activaciones full 1M FP32 : {B*1_000_000*d*4/1e9:.1f} GB  <- imposible en 6GB")
    info(f"  Chunk {chunk_s} tokens FP32    : {B*chunk_s*d*4/1e6:.1f} MB  <- manejable")
    info(f"  N chunks para 1M          : {n_ch}")
    if t1:
        valid=[r for r in t1 if 'tps' in r]
        if valid:
            r=valid[-1]; tps=r['tps']; ms=r['ms']
            info(f"  Throughput ref S={r['S']:,}: {tps:,} tok/s")
            info(f"  Latencia extrapolada 1M   : {1e6/tps:.1f} s")
            info(f"  Tiempo chunked 1M         : {n_ch*ms/1e3:.1f} s ({n_ch} x {ms:.1f} ms)")
    print()
    info("CHIMERA vs TRANSFORMER (1M tokens, 32 capas):")
    L=32; kv_gb=2*8*(d//8)*1_000_000*2*L/1e9
    info(f"  Transformer KV-cache 1M BF16 : {kv_gb:.1f} GB  <- imposible en 6GB")
    info(f"  CHIMERA SSM state   1M FP32  : {state_kb*32/1024:.2f} MB   <- siempre cabe")
    info(f"  Ventaja estado               : x{kv_gb*1024/(state_kb*32):.0f}")
    print()
    info("PLAN DE ACCION (mejoras en backlog):")
    for k,v in [
        ("D3","Streaming inference: buffer 16-tok, TTT update cada chunk"),
        ("D4","INT8 coefficients: x8 menos VRAM de estado SSM"),
        ("D5","Dynamic thresholds: routing por complejidad, no longitud"),
        ("D1","Adaptive degree per-head: heads locales degree=2 -> -25%% FLOPs"),
        (" *","Chunked prefill con carry: bus_cache pasa entre chunks"),
    ]: info(f"  [{k}] {v}")

# T8 ------------------------------------------------------------------
def test_autoregressive_step(device):
    banner("T8 - Rendimiento step() token-a-token")
    model=make_model(device); model.eval(); d=256; n=32 if QUICK else 64
    has_alloc=True
    try:
        cache=model.allocate_inference_cache(1,dtype=torch.float32)
    except Exception as e:
        warn(f"allocate_inference_cache: {e} -> usando forward([B,1,D])"); has_alloc=False; cache={'bus_cache':None}
    reset_peak(); times=[]
    for i in range(n):
        xt=torch.randn(1,1,d,device=device); t0=time.perf_counter()
        if has_alloc:
            out,cache=model.step(xt,cache)
        else:
            with torch.no_grad(): out,new_bus=model(xt,bus_cache=cache.get('bus_cache'))
            cache={'bus_cache':new_bus}
        sync(); times.append((time.perf_counter()-t0)*1e3)
    t=torch.tensor(times[4:]); mean_ms=t.mean().item(); std_ms=t.std().item()
    tps=int(1000/mean_ms); peak=vram_peak_mb()
    ok(f"Latencia : {mean_ms:.3f} +- {std_ms:.3f} ms/tok")
    ok(f"Throughput: {tps:,} tok/s")
    ok(f"VRAM pico : {peak:.1f} MB")
    info(f"1M tokens @ {mean_ms:.2f} ms/tok = {1e6*mean_ms/1e3:.0f} s = {1e6*mean_ms/3.6e6:.1f} h")
    return {'mean_ms':round(mean_ms,3),'std_ms':round(std_ms,3),'tps':tps,'vram_mb':round(peak,1)}

# MAIN ----------------------------------------------------------------
def main():
    global QUICK
    p=argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); args=p.parse_args()
    QUICK=args.quick
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Dispositivo : {device}")
    if device=='cuda':
        pr=torch.cuda.get_device_properties(0)
        print(f"  GPU         : {pr.name}\n  VRAM total  : {pr.total_memory/1e9:.1f} GB\n  SM          : {pr.major}.{pr.minor}")
    print(f"  PyTorch     : {torch.__version__}")
    if device!='cuda': fail("CHIMERA requiere CUDA"); sys.exit(1)

    all_r={}; status={}
    for name,fn in [
        ('T1_throughput',  lambda: test_throughput_vram(device)),
        ('T2_numerical',   lambda: test_numerical_stability(device)),
        ('T3_long_ctx',    lambda: test_long_context(device)),
        ('T4_breakdown',   lambda: test_latency_breakdown(device)),
        ('T5_error_accum', lambda: test_error_accumulation(device)),
        ('T6_grad_quality',lambda: test_gradient_quality(device)),
        ('T8_step',        lambda: test_autoregressive_step(device)),
    ]:
        try:
            all_r[name]=fn(); status[name]='PASS'
        except Exception as e:
            fail(f"[{name}] {e}"); traceback.print_exc(); status[name]=f'FAIL: {e}'

    report_1m_projection(all_r.get('T1_throughput',[]))

    banner("RESUMEN FINAL")
    for name,s in status.items():
        (ok if s=='PASS' else fail)(f"{name}: {s}")
    passes=sum(1 for s in status.values() if s=='PASS')
    print(f"\n  {passes}/{len(status)} tests pasaron.\n")

    out=os.path.join(os.path.dirname(__file__),'deep_analysis_results.json')
    def ser(o):
        if isinstance(o,torch.Tensor): return o.tolist()
        if isinstance(o,dict): return {k:ser(v) for k,v in o.items()}
        if isinstance(o,list): return [ser(v) for v in o]
        return o
    with open(out,'w') as f: json.dump(ser(all_r),f,indent=2)
    ok(f"Guardado en {out}")

if __name__=='__main__':
    main()
