"""
turboquant.py — A faithful, readable NumPy implementation of the TurboQuant
KV-cache quantization scheme (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026,
arXiv:2504.19874).

This is a *reference / validation* implementation, not a production kernel.
Its job is to let us empirically check the paper's claims and to act as the
algorithmic spec we can later port to MLX/Metal for the `iris` local agent.

Pipeline (per head vector x in R^d):
  1. Separate norm:        s = ||x||,  u = x / s        (store s in fp16)
  2. Random rotation:      y = R u    (R orthogonal, data-oblivious)
       -> after rotation each coord of a unit vector ~ N(0, 1/d)
  3. Scalar quantize:      each coord with a Lloyd-Max codebook for N(0,1),
                           scaled by 1/sqrt(d)   (b bits/coord)
  4. (optional) QJL:       1-bit sign of a random projection of the residual,
                           used to de-bias inner-product estimates.

Dequant reverses 1-3. The rotation is the whole trick: it is *data-oblivious*
(R never depends on the data), so it works online with zero calibration.
"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------
# Lloyd-Max optimal scalar quantizer for a standard normal source.
# Computed numerically so the code is self-documenting (no magic tables).
# --------------------------------------------------------------------------
def lloyd_max_gaussian(bits: int, n_samples: int = 2_000_000, iters: int = 100,
                       seed: int = 0) -> np.ndarray:
    """Return the 2**bits optimal reconstruction levels for N(0,1)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples)
    k = 2 ** bits
    # init levels at empirical quantiles
    levels = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        # boundaries are midpoints between adjacent levels
        bnds = (levels[:-1] + levels[1:]) / 2.0
        idx = np.searchsorted(bnds, x)
        new = np.array([x[idx == j].mean() if np.any(idx == j) else levels[j]
                        for j in range(k)])
        if np.allclose(new, levels, atol=1e-7):
            levels = new
            break
        levels = new
    return np.sort(levels)


def _quantize_to_levels(vals: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Nearest-level indices for `vals` given sorted `levels`."""
    bnds = (levels[:-1] + levels[1:]) / 2.0
    return np.searchsorted(bnds, vals).astype(np.int32)


# --------------------------------------------------------------------------
# Random orthogonal rotation (data-oblivious).
# A full QR-based matrix is O(d^2); production should use a randomized
# Hadamard transform (sign flip + Walsh-Hadamard) for O(d log d). We use the
# dense matrix here for clarity/correctness.
# --------------------------------------------------------------------------
def random_rotation(d: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    # fix signs so Q is a proper, deterministic orthogonal matrix
    q *= np.sign(np.diag(r))
    return q


class TurboQuant:
    """TurboQuant quantizer for fixed head dimension `d`."""

    def __init__(self, d: int, bits: int, use_qjl: bool = False,
                 qjl_dims: int | None = None, seed: int = 1234):
        self.d = d
        self.bits = bits
        self.use_qjl = use_qjl
        self.R = random_rotation(d, seed)
        self.levels = lloyd_max_gaussian(bits) / np.sqrt(d)  # coords ~ N(0,1/d)
        # QJL: random +-1 projection matrix for the residual (1 bit per row)
        self.m = qjl_dims if qjl_dims is not None else d
        if use_qjl:
            rng = np.random.default_rng(seed + 7)
            self.P = rng.standard_normal((self.m, d))  # JL projection

    # ---- encode / decode a batch X of shape (n, d) -----------------------
    def encode(self, X: np.ndarray):
        s = np.linalg.norm(X, axis=1, keepdims=True)          # (n,1) fp16 norm
        s = np.maximum(s, 1e-12)
        U = X / s
        Y = U @ self.R.T                                      # rotate
        idx = _quantize_to_levels(Y, self.levels)             # (n,d) int
        out = {"idx": idx, "norm": s.astype(np.float16)}
        if self.use_qjl:
            Yhat = self.levels[idx]
            resid = Y - Yhat
            out["qjl"] = (resid @ self.P.T >= 0)              # (n,m) bool, 1 bit
        return out

    def decode(self, enc) -> np.ndarray:
        Yhat = self.levels[enc["idx"]]
        Uhat = Yhat @ self.R                                  # inverse rotation (R orthogonal)
        return Uhat * enc["norm"].astype(np.float32)

    # ---- storage accounting ---------------------------------------------
    def bits_per_coord(self) -> float:
        # b bits/coord for the index + one fp16 norm amortised over d coords
        b = self.bits + 16.0 / self.d
        if self.use_qjl:
            b += self.m / self.d                              # 1 bit per qjl row
        return b

    def compression_vs_fp16(self) -> float:
        return 16.0 / self.bits_per_coord()


# --------------------------------------------------------------------------
# Asymmetric K/V wrapper: keys are attention-critical (quantize gently),
# values tolerate aggressive compression. This is the most robust, community-
# agreed practical win from the TurboQuant ecosystem.
# --------------------------------------------------------------------------
class AsymmetricKV:
    def __init__(self, d: int, k_bits: int, v_bits: int, seed: int = 1234):
        self.kq = TurboQuant(d, k_bits, seed=seed)
        self.vq = TurboQuant(d, v_bits, seed=seed + 1)

    def roundtrip_keys(self, K):
        return self.kq.decode(self.kq.encode(K))

    def roundtrip_values(self, V):
        return self.vq.decode(self.vq.encode(V))

    def effective_bits_per_coord(self):
        return 0.5 * (self.kq.bits_per_coord() + self.vq.bits_per_coord())
