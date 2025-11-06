import numpy as np

# Parameters
S0, K, T, r, sigma = 50.0, 40.0, 1.0, 0.05, 0.2
n = 100_000
rng = np.random.default_rng(42)

def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))

Z  = rng.standard_normal(n)
ST = S0 * np.exp((r - 0.5*sigma**2) * T + sigma * np.sqrt(T) * Z)

Xi = np.exp(-r*T) * np.maximum(0.0, ST - K)
Yi = np.exp(-r*T) * (ST - S0)
muY = S0 * (1.0 - np.exp(-r*T))                  

# Plain
plain_option_price = Xi.mean()
plain_SE = se(Xi)
print(f"Plain Estimated Call Option Price: {plain_option_price:.4f}")
print(f"Plain Standard Error: {plain_SE:.4f}")

# Control b=1
b1 = 1.0
H1 = Xi - b1 * (Yi - muY)
price_b1 = H1.mean()
se_b1    = se(H1)
print(f"Control Variate (b = 1) Estimated Call Option Price: {price_b1:.4f}")
print(f"Control Variate (b = 1) Standard Error: {se_b1:.4f}")

# Control b*
cov = np.cov(Xi, Yi, ddof=1)
b_star = cov[0, 1] / cov[1, 1]
H_star = Xi - b_star * (Yi - muY)
price_bstar = H_star.mean()
se_bstar    = se(H_star)

print(f"Control Variate (b* = Cov/Var = {b_star:.6f}) Estimated Call Option Price: {price_bstar:.4f}")
print(f"Control Variate (b* = Cov/Var = {b_star:.6f}) Standard Error: {se_bstar:.4f}")
