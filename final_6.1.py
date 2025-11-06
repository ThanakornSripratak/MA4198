import numpy as np

# Parameters
S0 = 50       
K = 50         # Strike price
T = 1.0        # Time to maturity (in years)
r = 0.05       # Risk-free rate
sigma = 0.2    # Volatility
n = 100000     # Number of Monte Carlo simulations

Z = np.random.randn(n)  
# Z ~ N(0,1)

# Simulate 
Si = S0 * np.exp((r - 0.5 * sigma**2)*T + (sigma * np.sqrt(T)*Z))
Xi = np.exp(-r*T) * np.maximum(0, Si-K)
Si_anti = S0 * np.exp((r - 0.5 * sigma**2)*T - (sigma * np.sqrt(T)*Z))
Yi = np.exp(-r*T) * np.maximum(0, Si_anti-K)

# Plain
plain_option_price = np.mean(Xi)
plain_SE = np.sqrt(np.sum((Xi**2 - plain_option_price**2)) / (n * (n - 1)))
print(f"Plain Estimated Call Option Price: {plain_option_price:.4f}")
print(f"Plain Standard Error: {plain_SE:.4f}")

# Antithetic
Zi = 0.5 * (Xi + Yi)
antithetic_option_price = np.mean(Zi)
antithetic_SE = np.std(Zi, ddof=1) / np.sqrt(n)

# Output
print(f"Antithetic Estimated Call Option Price: {antithetic_option_price:.4f}")
print(f"Antithetic Standard Error: {antithetic_SE:.4f}")