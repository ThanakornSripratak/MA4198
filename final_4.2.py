import numpy as np

def asian_call_mc(S0=50.0, K=50.0, r=0.05, sigma=0.2, T=1.0, m=12, n=10_000, seed=123):
    rng = np.random.default_rng(seed)
    dt    = T / m
    drift = (r - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    # simulate paths (vectorized)
    Z = rng.standard_normal((n, m))
    S_paths = S0 * np.exp(np.cumsum(drift + vol * Z, axis=1))  # shape (n,m)
    S_avg = S_paths.mean(axis=1)

    disc = np.exp(-r*T)
    X = disc * np.maximum(S_avg - K, 0.0)

    vhat = X.mean()
    SE = np.sqrt((np.sum(X**2) - n * vhat**2) / (n * (n - 1)))
    return vhat, SE

# Parameters
S0, T, r, sigma, m = 50.0, 1.0, 0.05, 0.2, 12
Ks = [40, 50, 60]

def run_and_print(n):
    out = []
    for K in Ks:
        v, se = asian_call_mc(S0=S0, K=K, r=r, sigma=sigma, T=T, m=m, n=n, seed=123)
        out.append((K, v, se))
    return out

res_2500  = run_and_print(2500)
res_10000 = run_and_print(10000)

# Output
for label, res in [("n=2500", res_2500), ("n=10000", res_10000)]:
    print(label)
    for K, v, se in res:
        print(f"  K={K:>2}: MC={v:.4f}, SE={se:.4f}")
