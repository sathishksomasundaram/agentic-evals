"""Spike 0004 — POC NumPy reference vs rachittshah port: encode/decode round-trip.

Per spike 0003's finding: all bit widths produce identical degenerate output during
real generation. Two hypotheses to disambiguate:

  (a) Port has an ALGORITHM bug — math diverges from POC reference.
  (b) Port math is fine; bug is in cache/integration with mlx-lm.

This spike answers (a). Method: identical synthetic K vectors through POC's
`TurboQuant.encode/decode` AND port's `PolarQuant.quantize_and_reconstruct`.
Compare reconstruction error magnitudes (different random rotations expected,
so equal error magnitude — not equal vectors — is the success criterion).

Run with:
    uv run agentic-evals exp-001 poc-vs-port-roundtrip
"""

# ruff: noqa: I001
# Reason: imports here are intentionally split by sys.path manipulation (POC reference
# lives outside the import path). Standard isort grouping would silently break the script.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

# The POC reference lives under spike 0001's turboquant-poc/. Add it to sys.path
# for direct import (Rule 6 — sys.path hack is appropriate inside a spike script).
POC_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "spikes"
    / "0001-inference-knobs-landscape"
    / "turboquant-poc"
)
sys.path.insert(0, str(POC_DIR))
# isort: off  — sys.path manipulation above means import order matters
from turboquant import TurboQuant  # type: ignore[import-not-found]  # noqa: E402

from mlx_turboquant.polar_quant import (  # type: ignore[import-untyped]  # noqa: E402
    PolarQuant,
)

# isort: on

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0004-port-vs-poc-roundtrip"
)


def make_kv(n_ctx: int, d: int, n_outliers: int = 4, seed: int = 0) -> np.ndarray:
    """Synthetic KV cache with outlier channels — same generator as the POC benchmark."""
    rng = np.random.default_rng(seed)
    scale = np.ones(d)
    scale[rng.choice(d, n_outliers, replace=False)] = 6.0
    return (rng.standard_normal((n_ctx, d)) * scale).astype(np.float32)


def reconstruction_metrics(
    original: np.ndarray, reconstructed: np.ndarray, label: str
) -> dict[str, Any]:
    err = original - reconstructed
    mse = float(np.mean(err**2))
    max_abs = float(np.max(np.abs(err)))
    cos = float(
        np.mean(
            np.sum(original * reconstructed, axis=1)
            / (np.linalg.norm(original, axis=1) * np.linalg.norm(reconstructed, axis=1) + 1e-12)
        )
    )
    return {
        "label": label,
        "mse": round(mse, 6),
        "max_abs": round(max_abs, 4),
        "cos": round(cos, 5),
    }


def main() -> None:
    n_ctx, d = 128, 128  # Llama-style head_dim; matches POC defaults
    K = make_kv(n_ctx, d, seed=0)
    print(f"K shape: {K.shape}, dtype: {K.dtype}")
    print(f"K stats: mean={K.mean():.3f}, std={K.std():.3f}, max|K|={np.max(np.abs(K)):.3f}")

    results: dict[str, Any] = {
        "n_ctx": n_ctx,
        "d": d,
        "K_stats": {
            "mean": float(K.mean()),
            "std": float(K.std()),
            "max_abs": float(np.max(np.abs(K))),
        },
        "by_bits": {},
    }

    for bits in (4, 3, 2):
        print(f"\n=== bits={bits} ===")

        # POC NumPy reference
        tq_poc = TurboQuant(d=d, bits=bits, seed=1234)
        enc = tq_poc.encode(K)
        K_poc = tq_poc.decode(enc)
        m_poc = reconstruction_metrics(K, K_poc, f"POC_bits{bits}")
        print(
            f"  POC : mse={m_poc['mse']:.4e}  max|err|={m_poc['max_abs']:.3f}  cos={m_poc['cos']}"
        )

        # Port: MLX PolarQuant.
        # NOTE: quantize_and_reconstruct returns (reconstructed, indices, norms),
        # NOT (indices, norms, reconstructed). Initial debug run had this reversed
        # and produced catastrophic-looking MSE — that was the spike script's bug,
        # not the port's bug. See REPORT.md "Iteration log" for details.
        pq_port = PolarQuant(bits=bits, dim=d, seed=42)
        K_port_mx, indices, norms = pq_port.quantize_and_reconstruct(mx.array(K))
        K_port = np.array(K_port_mx)
        m_port = reconstruction_metrics(K, K_port, f"port_bits{bits}")
        pm = m_port
        print(f"  port: mse={pm['mse']:.4e}  max|err|={pm['max_abs']:.3f}  cos={pm['cos']}")

        # Different random rotations → identical vectors NOT expected; equal
        # error magnitude IS expected for correctly-implemented algorithms.
        ratio_mse = m_port["mse"] / max(m_poc["mse"], 1e-12)
        cos_delta = m_port["cos"] - m_poc["cos"]
        print(f"  ratio port/POC mse = {ratio_mse:.2f}x   port-cos − POC-cos = {cos_delta:+.5f}")

        results["by_bits"][str(bits)] = {
            "poc": m_poc,
            "port": m_port,
            "ratio_port_to_poc_mse": round(ratio_mse, 3),
            "cos_delta_port_minus_poc": round(cos_delta, 5),
        }

    # Verdict heuristic — within ~3x mse and cos within 0.01 = algorithm matches.
    print("\n=== Verdict ===")
    for bits in (4, 3, 2):
        r = results["by_bits"][str(bits)]
        ratio = r["ratio_port_to_poc_mse"]
        cos_d = r["cos_delta_port_minus_poc"]
        ok = ratio < 3.0 and abs(cos_d) < 0.01
        verdict = "MATCH (algorithm parity)" if ok else "DIVERGE (port may have algo bug)"
        print(f"  bits={bits}: {verdict}  [mse-ratio={ratio:.2f}x, cos-delta={cos_d:+.5f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "raw-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
