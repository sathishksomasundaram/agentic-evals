"""
benchmark.py — Empirically validate the TurboQuant claims that matter for a
local agent like `iris`:

  Claim A: ~quality-neutral at 3.5 bits/coord, marginal degradation at 2.5.
  Claim B: ~5x KV-cache compression vs fp16.
  Claim C: asymmetric K/V (gentle K, aggressive V) preserves attention.

The decisive metric for an LLM is NOT vector reconstruction error — it is
whether the attention distribution softmax(QK^T/sqrt(d)) and the resulting
context vector survive quantization. We measure both.
"""

import numpy as np
from turboquant import TurboQuant, AsymmetricKV


def make_kv(n_ctx, d, n_outliers=4, seed=0):
    """Synthetic KV cache with a few high-variance 'outlier' channels,
    mimicking the heavy-tailed per-channel structure of real LLM caches."""
    rng = np.random.default_rng(seed)
    scale = np.ones(d)
    scale[rng.choice(d, n_outliers, replace=False)] = 6.0   # outlier channels
    K = rng.standard_normal((n_ctx, d)) * scale
    V = rng.standard_normal((n_ctx, d)) * scale
    return K.astype(np.float32), V.astype(np.float32)


def attention(Q, K, V):
    d = Q.shape[1]
    logits = Q @ K.T / np.sqrt(d)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p, p @ V


def kl(p, q, eps=1e-9):
    p = p + eps; q = q + eps
    return (p * np.log(p / q)).sum(axis=1).mean()


def make_retrieval_queries(K, n_query, noise=0.35, seed=1):
    """Each query is a noisy copy of one random key -> peaked attention,
    so we can measure needle-in-a-haystack retrieval (the paper's metric)."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, K.shape[0], n_query)
    Q = K[targets] + noise * rng.standard_normal((n_query, K.shape[1])) \
        * np.linalg.norm(K, axis=1).mean()
    return Q.astype(np.float32), targets


def evaluate(K, V, Q, targets, Kq, Vq):
    d = K.shape[1]
    p_ref, o_ref = attention(Q, K, V)
    p_q, _ = attention(Q, Kq, V)
    _, o_qq = attention(Q, Kq, Vq)
    # needle retrieval: did the top-attended key stay the same as fp16?
    needle = np.mean(p_q.argmax(1) == p_ref.argmax(1)) * 100
    kld = kl(p_ref, p_q)
    out_cos = np.mean(np.sum(o_ref * o_qq, 1) /
                      (np.linalg.norm(o_ref, axis=1) *
                       np.linalg.norm(o_qq, axis=1) + 1e-12))
    kcos = np.mean(np.sum(K * Kq, 1) /
                   (np.linalg.norm(K, axis=1) * np.linalg.norm(Kq, axis=1)))
    return kcos, kld, needle, out_cos


def run(d=128, n_ctx=2048, n_query=512, seed=0):
    K, V = make_kv(n_ctx, d, seed=seed)
    Q, targets = make_retrieval_queries(K, n_query, seed=seed + 1)

    print(f"\n=== head_dim={d}, context={n_ctx} (needle retrieval) ===")
    print(f"{'scheme':<20}{'bits/cd':>8}{'compress':>10}"
          f"{'K cos':>9}{'attn KL':>10}{'needle%':>9}{'out cos':>9}")
    print("-" * 75)

    rows = []
    for bits in [2, 3, 4]:
        tq = TurboQuant(d, bits, seed=seed)
        Kq, Vq = tq.decode(tq.encode(K)), tq.decode(tq.encode(V))
        kcos, kld, needle, ocos = evaluate(K, V, Q, targets, Kq, Vq)
        bpc, comp = tq.bits_per_coord(), tq.compression_vs_fp16()
        print(f"{'symmetric '+str(bits)+'b':<20}{bpc:>8.2f}{comp:>9.2f}x"
              f"{kcos:>9.4f}{kld:>10.5f}{needle:>8.1f}%{ocos:>9.4f}")
        rows.append((f"sym {bits}b", bpc, comp, kcos, kld, needle, ocos))

    for kb, vb, label in [(4, 2, "K4/V2"), (4, 3, "K4/V3"), (3, 2, "K3/V2")]:
        akv = AsymmetricKV(d, kb, vb, seed=seed)
        Kq, Vq = akv.roundtrip_keys(K), akv.roundtrip_values(V)
        kcos, kld, needle, ocos = evaluate(K, V, Q, targets, Kq, Vq)
        bpc = akv.effective_bits_per_coord(); comp = 16.0 / bpc
        print(f"{'asym '+label:<20}{bpc:>8.2f}{comp:>9.2f}x"
              f"{kcos:>9.4f}{kld:>10.5f}{needle:>8.1f}%{ocos:>9.4f}")
        rows.append((f"asym {label}", bpc, comp, kcos, kld, needle, ocos))

    return rows


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    run(d=128, n_ctx=2048)     # Llama-style head_dim
    run(d=256, n_ctx=2048)     # Gemma-style head_dim (paper: larger d helps)
