"""Collapse proof for the 'Binary VSA Hyperlayer' proposed as an FFN replacement.

Claim: with frozen bipolar T1,T2 and a fixed permutation, every step between the
first sign() and the trainable decode is inert. The whole 'multi-step VSA
program' equals a single frozen random binary projection + trainable linear
readout:   y = sign(x @ P) @ W_eff + b.

If true, T1/T2/permute/inner-binarize add ZERO capacity over a random-features
FFN, and the layer cannot out-represent a learned SwiGLU. We show bit-identity.
"""
import numpy as np

rng = np.random.default_rng(0)
B, S, d, D, shift = 2, 7, 64, 10240, 42

def bip(x): return np.where(x >= 0, 1.0, -1.0)

x     = rng.standard_normal((B, S, d)).astype(np.float32)
P     = bip(rng.standard_normal((d, D)))          # frozen encoder
T1    = bip(rng.standard_normal((D,)))            # frozen rule vec
T2    = bip(rng.standard_normal((D,)))            # frozen rule vec
W_dec = rng.standard_normal((D, d)).astype(np.float32) * (0.02 / D**0.5)
b     = rng.standard_normal((d,)).astype(np.float32)

# ---- faithful forward, exactly as the document specifies ----
def faithful(x):
    X_hd      = bip(x @ P)                         # encode + majority gate
    H1        = X_hd * T1                          # bind T1
    H1_active = bip(H1)                            # "denoise" (claimed nonlinearity)
    H2        = np.roll(H1_active, shift, axis=-1) * T2   # permute + bind T2
    Y_hd      = bip(H2)                            # final majority gate
    return Y_hd @ W_dec + b, X_hd, H1, H1_active, H2, Y_hd

y_faith, X_hd, H1, H1_active, H2, Y_hd = faithful(x)

# ---- show the two inner binarize() calls are no-ops on already-bipolar data ----
noop1 = np.abs(H1_active - H1).max()              # bip(H1) == H1 ?
noop2 = np.abs(Y_hd - H2).max()                   # bip(H2) == H2 ?

# ---- collapsed forward: single frozen projection + one linear readout ----
C       = np.roll(T1, shift) * T2                 # frozen +-1 constant
rows    = C[:, None] * W_dec                      # fold elementwise C into decode
W_eff   = np.roll(rows, -shift, axis=0)           # fold the permutation into rows
y_collapse = bip(x @ P) @ W_eff + b

diff = np.abs(y_faith - y_collapse).max()

print(f"inner binarize #1 is a no-op (max|.|)      : {noop1:.3e}")
print(f"inner binarize #2 is a no-op (max|.|)      : {noop2:.3e}")
print(f"faithful vs collapsed  max_abs_diff        : {diff:.3e}")
print()
print("=> the multi-step VSA core == sign(x@P_frozen) @ W_eff + b")
print("   T1, T2, permute, inner-binarize add zero capacity.")
print(f"   hidden features are FROZEN random binary (D={D}); only the readout trains.")
assert noop1 == 0.0 and noop2 == 0.0 and diff < 1e-4, "collapse disproven!"
print("\nPASS: collapse proven bit-identical.")
