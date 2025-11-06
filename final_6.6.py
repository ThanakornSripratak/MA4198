import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from math import log, sqrt, exp

def bs_call_price(S0, K, T, r, sigma):
    d1 = (log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

def payoff_from_U(U, S0, K, T, r, sigma):
    Z = norm.ppf(U)                     # Φ^{-1}(U)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma * sqrt(T) * Z)
    return np.exp(-r*T) * np.maximum(ST - K, 0.0)

# Stratifed Sampling
def stratified_call(S0, K, T, r, sigma, n, k, rng):
    widths = np.full(k, 1.0/k)
    nh = np.floor(n * widths).astype(int)
    remainder = n - nh.sum()
    if remainder > 0:
        nh[:remainder] += 1

    stratum_means = np.zeros(k)
    stratum_vars  = np.zeros(k)

    for h in range(k):
        m = nh[h]
        if m == 0:
            continue
        U = (h + rng.random(m)) / k
        Xi = payoff_from_U(U, S0, K, T, r, sigma)
        stratum_means[h] = Xi.mean()
        stratum_vars[h]  = Xi.var(ddof=1) if m > 1 else 0.0

    w = widths
    est = np.sum(w * stratum_means)
    var = np.sum((w**2) * (stratum_vars / np.maximum(nh, 1)))
    se  = np.sqrt(var)
    return est, se

if __name__ == "__main__":
    S0, r, sigma, T = 50.0, 0.05, 0.2, 1.0
    Ks = [40, 50, 60]
    strata_list = [25, 100]
    n = 100_000
    rng = default_rng(42)

    print("Stratified sampling for European Call (Example 6.6)\n")
    print(f"Params: S0={S0}, r={r}, sigma={sigma}, T={T}, n={n}")
    print("-" * 78)
    print(f"{'K':>4} | {'k (strata)':>10} | {'True (BS)':>10} | {'Estimate':>10} | {'S.E.':>8} | {'95% CI':>23}")
    print("-" * 78)

    for K in Ks:
        true_price = bs_call_price(S0, K, T, r, sigma)
        for k in strata_list:
            est, se = stratified_call(S0, K, T, r, sigma, n, k, rng)
            lo, hi = est - 1.96*se, est + 1.96*se
            print(f"{K:>4} | {k:>10} | {true_price:>10.4f} | {est:>10.4f} | {se:>8.4f} | [{lo:>8.4f}, {hi:>8.4f}]")

    print("-" * 78)