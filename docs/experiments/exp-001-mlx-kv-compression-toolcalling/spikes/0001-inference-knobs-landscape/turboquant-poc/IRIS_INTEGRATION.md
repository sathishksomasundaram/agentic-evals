# TurboQuant → `iris` integration brief

## What this POC proves (and doesn't)
- **Proves**: the compression math is exact (5.12x @ d=128, 5.22x @ d=256 at
  3-bit), the data-oblivious random-rotation + scalar-quant pipeline works with
  zero calibration, and **asymmetric K/V dominates** — `K4/V2` matches full
  4-bit attention quality at the memory cost of 3-bit.
- **Doesn't prove**: production token/s (needs a Metal kernel, not NumPy), or
  perplexity on real text. This is the algorithm spec + a quality harness, not
  an inference engine.

## Where it lives in iris
TurboQuant operates **inside the inference runtime**, not in your TS/Nest layer.
The KV cache never reaches Node. So iris's job is orchestration + policy:

```
iris (Next.js / NestJS)              inference runtime (the only place KV lives)
  ├─ model registry / selector  ──▶  mlx-lm fork  OR  llama.cpp fork
  ├─ kv-quant policy (k_bits,        └─ TurboQuant KV cache  ◀── port of this POC
  │   v_bits, protected layers)
  ├─ context-length budgeter
  └─ telemetry (tok/s, ctx, RAM)
```

Recommended default policy for a Mac local agent: **k_bits=4, v_bits=2or3,
leave first/last 2 layers in fp16** (the "boundary" trick the community uses to
recover precision). Expose it as a per-model config, not a global flag.

## Concrete next steps
1. **Pick the runtime.** For Apple Silicon the two live options are an
   `mlx-lm` drop-in (easiest to integrate with a Mac-native agent) or a
   `llama.cpp` TurboQuant fork (more mature per community reports). Neither is
   official Google code — see security note.
2. **Port the codebook + rotation** from `turboquant.py` to a fused Metal kernel
   (randomized Hadamard transform instead of the dense QR matrix here — O(d log d)).
3. **Wire the policy object** into your model registry and surface ctx-length
   gains in iris telemetry (the real UX win is longer context, not a smaller model).
4. **Validate on YOUR workload** with a real needle-in-a-haystack + a perplexity
   check before trusting any "zero loss" claim.

## Security / privacy notes (your wheelhouse)
- **Supply chain**: every TurboQuant implementation today is community code with
  no official Google release. A KV-quant kernel sits on the hottest path of your
  model and sees every token. Pin commits, build from source, review the Metal/
  CUDA kernel, and sandbox the runtime. Treat it as untrusted until audited.
- **Privacy upside**: this is purely local memory compression — no data leaves
  the device, fully consistent with iris's privacy-first thesis. It strengthens
  the local-only story (longer context without cloud offload).
- **Failure mode to monitor**: aggressive V/K bits silently degrade retrieval
  rather than crashing. Keep the needle metric in CI so a quality regression in
  a kernel update is caught, not shipped.
