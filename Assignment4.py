import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Parameters
sensor_reading = 34
sensor_std = 20
prior_min, prior_max = 0, 182
true_value = 50  # True fuel level for positive control
max_iterations = 100000  # Maximum iterations for MH
max_samples = 100000  # Maximum samples for BMC
x_values = np.linspace(0, 182, 1000)  # Range for estimating mode

# Likelihood function (Gaussian)
def likelihood(x, sensor_reading, sensor_std):
    return np.exp(-0.5 * ((x - sensor_reading) / sensor_std) ** 2)

# Prior function (uniform)
def prior(x, prior_min, prior_max):
    return 1 if prior_min <= x <= prior_max else 0

# Posterior function (proportional to likelihood * prior)
def posterior(x, sensor_reading, sensor_std, prior_min, prior_max):
    return likelihood(x, sensor_reading, sensor_std) * prior(x, prior_min, prior_max)

# Metropolis-Hastings algorithm
def metropolis_hastings(sensor_reading, sensor_std, prior_min, prior_max, num_iterations):
    samples = []
    current = sensor_reading  # Start at the sensor reading
    for _ in range(num_iterations):
        proposal = current + np.random.normal(0, 5)  # Proposal distribution
        acceptance_prob = min(1, posterior(proposal, sensor_reading, sensor_std, prior_min, prior_max) / 
                             posterior(current, sensor_reading, sensor_std, prior_min, prior_max))
        if np.random.rand() < acceptance_prob:
            current = proposal
        samples.append(current)
    return samples

# Bayes Monte Carlo algorithm
def bayes_monte_carlo(sensor_reading, sensor_std, prior_min, prior_max, num_samples):
    samples = np.random.uniform(prior_min, prior_max, num_samples)
    weights = likelihood(samples, sensor_reading, sensor_std)
    return samples, weights

# Evaluate MH convergence using KDE
def evaluate_mh_convergence(sensor_reading, sensor_std, prior_min, prior_max, true_value, max_iterations):
    errors = []
    modes = []
    iteration_counts = range(1000, max_iterations + 1, 1000)
    
    for num_iterations in iteration_counts:
        samples = metropolis_hastings(sensor_reading, sensor_std, prior_min, prior_max, num_iterations)
        kde = gaussian_kde(samples)
        mode = x_values[np.argmax(kde(x_values))]
        error = abs(mode - true_value)
        modes.append(mode)
        errors.append(error)
        print(f"MH: Iterations = {num_iterations}, Estimated Mode = {mode:.2f}, Error = {error:.2f}")
    
    return iteration_counts, modes, errors

# Evaluate BMC convergence using KDE
def evaluate_bmc_convergence(sensor_reading, sensor_std, prior_min, prior_max, true_value, max_samples):
    errors = []
    modes = []
    sample_counts = range(1000, max_samples + 1, 1000)
    
    for num_samples in sample_counts:
        samples, weights = bayes_monte_carlo(sensor_reading, sensor_std, prior_min, prior_max, num_samples)
        kde = gaussian_kde(samples, weights=weights)
        mode = x_values[np.argmax(kde(x_values))]
        error = abs(mode - true_value)
        modes.append(mode)
        errors.append(error)
        print(f"BMC: Samples = {num_samples}, Estimated Mode = {mode:.2f}, Error = {error:.2f}")
    
    return sample_counts, modes, errors

# Run evaluations
mh_iterations, mh_modes, mh_errors = evaluate_mh_convergence(sensor_reading, sensor_std, prior_min, prior_max, true_value, max_iterations)
bmc_samples, bmc_modes, bmc_errors = evaluate_bmc_convergence(sensor_reading, sensor_std, prior_min, prior_max, true_value, max_samples)

# Plot mode convergence
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mh_iterations, mh_modes, label="Metropolis-Hastings", marker="o")
plt.axhline(true_value, color="red", linestyle="--", label="True Value")
plt.xlabel("Number of Iterations")
plt.ylabel("Estimated Mode")
plt.title("MH Mode Convergence")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bmc_samples, bmc_modes, label="Bayes Monte Carlo", marker="o")
plt.axhline(true_value, color="red", linestyle="--", label="True Value")
plt.xlabel("Number of Samples")
plt.ylabel("Estimated Mode")
plt.title("BMC Mode Convergence")
plt.legend()

plt.tight_layout()
plt.show()

# Plot error convergence
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(mh_iterations, mh_errors, label="Metropolis-Hastings", marker="o")
plt.xlabel("Number of Iterations")
plt.ylabel("Error (|Estimated Mode - True Value|)")
plt.title("MH Error Convergence")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bmc_samples, bmc_errors, label="Bayes Monte Carlo", marker="o")
plt.xlabel("Number of Samples")
plt.ylabel("Error (|Estimated Mode - True Value|)")
plt.title("BMC Error Convergence")
plt.legend()

plt.tight_layout()
plt.show()