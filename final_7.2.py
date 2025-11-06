import numpy as np
from math import exp, sqrt, log
from dataclasses import dataclass

S0   = 50.0
r    = 0.05
sigma= 0.2
T    = 1.0
n    = 10_000
Ks   = [50, 60, 80, 100, 120]

disc  = exp(-r*T)
mu    = (r - 0.5*sigma*sigma) * T
vol   = sigma * sqrt(T)

rng = np.random.default_rng(123456) 

def bs_call_price(S0, K, T, r, sigma):
    from math import erf
    def norm_cdf(x):
        return 0.5*(1.0 + erf(x / sqrt(2.0)))
    d1 = (log(S0/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S0*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)

# Plain
def call_plain_mc(K, n, rng):
    Z  = rng.standard_normal(n)
    ST = S0 * np.exp(mu + vol*Z)
    H  = disc * np.maximum(ST - K, 0.0)
    vhat = H.mean()
    # unbiased SE: sqrt( (sum H^2 - n vhat^2) / (n(n-1)) )
    se   = np.sqrt((np.sum(H*H) - n*vhat*vhat) / (n*(n-1)))
    return vhat, se

# Solve for x* 
def F_equation(x, K):
    return S0 * np.exp(mu + vol*x) * (vol - x) + disc * K * x

def solve_xstar_bisect(K, x_lo=1e-8, x_hi_initial=2.0, max_expand=12, tol=1e-10, itmax=200):
    x_hi = x_hi_initial
    f_lo = F_equation(x_lo, K)
    f_hi = F_equation(x_hi, K)
    expands = 0
    while f_lo * f_hi > 0 and expands < max_expand:
        x_hi *= 1.5
        f_hi = F_equation(x_hi, K)
        expands += 1
    if f_lo * f_hi > 0:
        x_hi = 10.0
        f_hi = F_equation(x_hi, K)
        if f_lo * f_hi > 0:
            raise RuntimeError("Error.")
    # Bisection
    a, b = x_lo, x_hi
    fa, fb = f_lo, f_hi
    for _ in range(itmax):
        m = 0.5*(a+b)
        fm = F_equation(m, K)
        if abs(fm) < tol or (b-a) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

# Importance Sampling
def call_is_mc(K, n, theta, rng):
    X  = rng.normal(loc=theta, scale=1.0, size=n)
    ST = S0 * np.exp(mu + vol*X)
    payoff = np.maximum(ST - K, 0.0)
    weights = np.exp(-theta*X + 0.5*theta*theta)
    H = disc * payoff * weights
    vhat = H.mean()
    se   = np.sqrt((np.sum(H*H) - n*vhat*vhat) / (n*(n-1)))
    return vhat, se

# Output
@dataclass
class RowPlain:
    K: float; theo: float; est: float; se: float; re: float

@dataclass
class RowIS:
    K: float; theo: float; est: float; se: float; re: float; xstar: float

plain_rows = []
is_rows = []

print("Table: Plain Monte Carlo for call option")
print("{:<6} {:>14} {:>14} {:>10} {:>8}".format("K","Theoretical","Estimate","S.E.","R.E.%"))
for K in Ks:
    theo = bs_call_price(S0, K, T, r, sigma)
    est, se = call_plain_mc(K, n, rng)
    re = 100.0 * se / theo
    plain_rows.append(RowPlain(K, theo, est, se, re))
    print("{:<6.0f} {:>14.4f} {:>14.4f} {:>10.4f} {:>8.2f}".format(K, theo, est, se, re))

print("\nTable: Importance Sampling for call option (mean-shift N(theta,1))")
print("{:<6} {:>14} {:>14} {:>10} {:>8} {:>10}".format("K","Theoretical","Estimate","S.E.","R.E.%","x*"))
for K in Ks:
    theo = bs_call_price(S0, K, T, r, sigma)
    xstar = solve_xstar_bisect(K)
    est, se = call_is_mc(K, n, xstar, rng)
    re = 100.0 * se / theo
    is_rows.append(RowIS(K, theo, est, se, re, xstar))
    print("{:<6.0f} {:>14.4f} {:>14.4f} {:>10.4f} {:>8.2f} {:>10.4f}".format(K, theo, est, se, re, xstar))
