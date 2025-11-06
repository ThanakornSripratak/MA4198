import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp

# BLS
def bs_call_price(S0, K, T, r, sigma):
    d1 = (log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

# Plain
def call_plain_mc(S0, K, T, r, sigma, n=100000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma * sqrt(T) * Z)
    payoff = np.exp(-r*T) * np.maximum(ST - K, 0.0)
    est = np.mean(payoff)
    se  = np.std(payoff, ddof=1) / sqrt(n)
    return est, se

# Combined
def call_antithetic_control_variate(S0, K, T, r, sigma, n=100000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if n % 2 != 0:
        n += 1

    m = n // 2
    Z = rng.standard_normal(m)
    drift = (r - 0.5*sigma**2)*T
    vol = sigma * sqrt(T)
    disc = exp(-r*T)

    ST1 = S0 * np.exp(drift + vol * Z)
    ST2 = S0 * np.exp(drift - vol * Z)
    Y1 = disc * np.maximum(ST1 - K, 0.0)
    Y2 = disc * np.maximum(ST2 - K, 0.0)
    X1 = disc * ST1
    X2 = disc * ST2
    muX = S0

    Ybar = 0.5 * (Y1 + Y2)
    Xbar = 0.5 * (X1 + X2)
    Xc = Xbar - np.mean(Xbar)
    Yc = Ybar - np.mean(Ybar)
    beta = np.mean(Xc * Yc) / np.mean(Xc**2)
    adj = Ybar - beta * (Xbar - muX)
    est = np.mean(adj)
    se  = np.std(adj, ddof=1) / sqrt(m)
    return est, se

if __name__ == "__main__":
    S0, K, T, r, sigma = 50.0, 50.0, 1.0, 0.05, 0.2
    n = 10000
    rng = np.random.default_rng(42)

    plain_est, plain_se = call_plain_mc(S0, K, T, r, sigma, n, rng)
    acv_est, acv_se = call_antithetic_control_variate(S0, K, T, r, sigma, n, rng)

    print(f"Plain Estimated Call Option Price: {plain_est:.4f}")
    print(f"Plain Standard Error: {plain_se:.4f}")
    print(f"Antithetic + Control Variate Estimated Call Option Price: {acv_est:.4f}")
    print(f"Antithetic + Control Variate Standard Error: {acv_se:.4f}")
