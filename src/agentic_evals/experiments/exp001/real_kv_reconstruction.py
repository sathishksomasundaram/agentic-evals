"""Spike 0004 follow-up — PolarQuant reconstruction on REAL Qwen-2.5 K/V.

Spike 0004 showed algorithm parity on synthetic data. Spike 0003 still failed
end-to-end. This script closes the loop: extract one layer's K/V from a real
prefill of Qwen-2.5-7B-Instruct, run those vectors through PolarQuant, and
compare to the synthetic-data result.

Result documented in spike 0004's REPORT.md (Part B).

Run with:
    uv run agentic-evals exp-001 real-kv-reconstruction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from mlx_turboquant.polar_quant import PolarQuant  # type: ignore[import-untyped]

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
LAYER_TO_INSPECT = 0
OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0004-port-vs-poc-roundtrip"
)


def _metrics(orig: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    err = orig - recon
    mse = float(np.mean(err**2))
    max_abs = float(np.max(np.abs(err)))
    cos = float(
        np.mean(
            np.sum(orig * recon, axis=1)
            / (np.linalg.norm(orig, axis=1) * np.linalg.norm(recon, axis=1) + 1e-12)
        )
    )
    return {"mse": round(mse, 6), "max_abs": round(max_abs, 4), "per_row_cos": round(cos, 4)}


def main() -> None:
    print(f"==> Loading {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)  # type: ignore[misc]

    prompt = "Write a poem about the ocean and the moon and the stars in the night sky. " * 5
    tokens = mx.array([tokenizer.encode(prompt)])
    print(f"==> Prompt tokens: {tokens.shape[1]}")

    # Use the standard cache to capture real K/V — no TurboQuant involved here
    num_layers = len(model.model.layers)
    caches = [KVCache() for _ in range(num_layers)]  # type: ignore[no-untyped-call]
    _ = model(tokens, cache=caches)
    mx.eval(caches[LAYER_TO_INSPECT].keys)
    mx.eval(caches[LAYER_TO_INSPECT].values)

    K = np.array(caches[LAYER_TO_INSPECT].keys).astype(np.float32)
    V = np.array(caches[LAYER_TO_INSPECT].values).astype(np.float32)
    print(f"\nLayer-{LAYER_TO_INSPECT} K shape: {K.shape}  V shape: {V.shape}")

    K_flat = K.reshape(-1, K.shape[-1])
    V_flat = V.reshape(-1, V.shape[-1])

    per_head_std: list[float] = [float(K[0, h].std()) for h in range(K.shape[1])]
    real_K_stats: dict[str, Any] = {
        "shape": list(K.shape),
        "mean": float(K_flat.mean()),
        "std": float(K_flat.std()),
        "max_abs": float(np.abs(K_flat).max()),
        "per_head_std": per_head_std,
    }
    real_V_stats = {
        "shape": list(V.shape),
        "mean": float(V_flat.mean()),
        "std": float(V_flat.std()),
        "max_abs": float(np.abs(V_flat).max()),
    }
    ks, vs = real_K_stats, real_V_stats
    print(f"\nReal K stats: mean={ks['mean']:.4f}  std={ks['std']:.3f}  max={ks['max_abs']:.2f}")
    print(f"Real V stats: mean={vs['mean']:.4f}  std={vs['std']:.4f}  max={vs['max_abs']:.3f}")
    rounded_per_head = [round(s, 2) for s in per_head_std]
    print(f"Per-head K std (GQA, {K.shape[1]} heads): {rounded_per_head}")

    d = K.shape[-1]
    pq = PolarQuant(bits=3, dim=d, seed=42)
    recon_K_mx, _, _ = pq.quantize_and_reconstruct(mx.array(K_flat))
    recon_V_mx, _, _ = pq.quantize_and_reconstruct(mx.array(V_flat))
    recon_K = np.array(recon_K_mx)
    recon_V = np.array(recon_V_mx)

    m_real_K = _metrics(K_flat, recon_K)
    m_real_V = _metrics(V_flat, recon_V)

    mk, mv = m_real_K, m_real_V
    print("\nReal-data reconstruction (PolarQuant bits=3):")
    print(f"  K: mse={mk['mse']:.4e}  max|err|={mk['max_abs']:.3f}  cos={mk['per_row_cos']:.4f}")
    print(f"  V: mse={mv['mse']:.4e}  max|err|={mv['max_abs']:.3f}  cos={mv['per_row_cos']:.4f}")
    print("\nFor comparison — synthetic data (spike 0004 bits=3): per-row cos = 0.98267")

    results: dict[str, Any] = {
        "model_id": MODEL_ID,
        "layer_inspected": LAYER_TO_INSPECT,
        "prompt_tokens": int(tokens.shape[1]),
        "real_K_stats": real_K_stats,
        "real_V_stats": real_V_stats,
        "polar_quant_bits": 3,
        "polar_quant_seed": 42,
        "reconstruction_real_K": m_real_K,
        "reconstruction_real_V": m_real_V,
        "reconstruction_synthetic_reference": {"per_row_cos": 0.98267, "mse": 0.0727},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "real-kv-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
