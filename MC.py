import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Task a
N = 100000  # Sample size for Monte Carlo simulation
samples = np.random.normal(0, 1, N)

# Analytical values
mean_analytical = 0
percentile_95_analytical = norm.ppf(0.95)

# Empirical values
mean_empirical = np.mean(samples)
percentile_95_empirical = np.percentile(samples, 95)

# Bootstrapping for uncertainty estimation
bootstrap_means = [np.mean(np.random.choice(samples, N, replace=True)) for _ in range(1000)]
bootstrap_percentiles = [np.percentile(np.random.choice(samples, N, replace=True), 95) for _ in range(1000)]

mean_uncertainty = np.std(bootstrap_means)
percentile_uncertainty = np.std(bootstrap_percentiles)

# Task b
N_pi = 1000000
x = np.random.uniform(-1, 1, N_pi)
y = np.random.uniform(-1, 1, N_pi)
inside_circle = x**2 + y**2 <= 1

pi_estimate = 4 * np.sum(inside_circle) / N_pi
pi_uncertainty = 4 * np.std(inside_circle) / np.sqrt(N_pi)

# Print Results
print("Task a: Normal Distribution Analysis")
print(f"Mean (Analytical): {mean_analytical}")
print(f"Mean (Empirical): {mean_empirical:.5f} ± {mean_uncertainty:.5f}")
print(f"95th Percentile (Analytical): {percentile_95_analytical:.5f}")
print(f"95th Percentile (Empirical): {percentile_95_empirical:.5f} ± {percentile_uncertainty:.5f}\n")

print("Task b: Estimation of pi using Monte Carlo")
print(f"Estimated pi: {pi_estimate:.5f} ± {pi_uncertainty:.5f}")

# Plotting Results
plt.figure(figsize=(10, 5))

# Figure 1: Histogram for normal distribution
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Samples')
x_vals = np.linspace(-4, 4, 1000)
plt.plot(x_vals, norm.pdf(x_vals), 'r-', lw=2, label='Analytical PDF')
plt.axvline(percentile_95_analytical, color='b', linestyle='--', label='95th Percentile (Analytical)')
plt.axvline(percentile_95_empirical, color='orange', linestyle='--', label='95th Percentile (Empirical)')
plt.title('Standard Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Figure 2: Scatter plot for Monte Carlo simulation of pi
plt.subplot(1, 2, 2)
plt.scatter(x[::1000], y[::1000], c=inside_circle[::1000], cmap='coolwarm', s=1, label='Points')
plt.title('Estimating π using Monte Carlo')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()

# Save and display figures
plt.tight_layout()
plt.savefig('results.png')
plt.show()
