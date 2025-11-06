import numpy as np
from math import log, sqrt, erf, exp

# Parameters
S0 = 50        # Initial stock price
K = 50         # Strike price
T = 1.0        # Time to maturity (in years)
r = 0.05       # Risk-free rate
sigma = 0.2    # Volatility
n = 100000     # Number of Monte Carlo simulations

def norm_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

# BLS
d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
bs_price = S0 * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

Z = np.random.randn(n)  # Z ~ N(0,1)

# Simulate
Y = S0 * np.exp((r - 0.5 * sigma**2)*T + (sigma * np.sqrt(T)*Z))
X = np.exp(-r*T) * np.maximum(0, Y-K)
option_price = np.mean(X)
standard_error = np.sqrt(np.sum((X**2 - option_price**2)) / (n * (n - 1)))

# Output
print(f"Theoretical (Black–Scholes) Price: {bs_price:.4f}")
print(f"Estimated Call Option Price: {option_price:.4f}")
print(f"Standard Error: {standard_error:.4f}")
