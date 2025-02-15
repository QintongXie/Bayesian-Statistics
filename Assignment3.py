import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Given parameters
fuel_capacity = 182  # liters
fuel_sensor_reading = 34  # liters
sensor_error_std = 20  # liters
fuel_consumption_rate = 18  # liters per hour
consumption_error_std = 2  # liters per hour

# 2. Initial PDF
x = np.linspace(-50, fuel_capacity, 1000)  # Extended range for negative values
pdf_fuel = stats.norm.pdf(x, fuel_sensor_reading, sensor_error_std)

plt.figure(figsize=(8,5))
plt.plot(x, pdf_fuel, label="Fuel Estimate (Sensor Only)")
plt.axvline(0, color='r', linestyle='--', label='Zero Fuel')
plt.xlabel("Fuel in Tank (liters)")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Initial Probability Density Function of Fuel Level")
plt.show()

# Expected Value and Most Likely Value (Mode) of Fuel Level
expected_fuel = fuel_sensor_reading  # Mean of normal distribution
mode_fuel = fuel_sensor_reading  # Mode is same as mean for Gaussian
prob_negative_fuel = stats.norm.cdf(0, fuel_sensor_reading, sensor_error_std)

print(f"Expected Fuel Level: {expected_fuel:.2f} liters")
print(f"Most Likely Fuel Level: {mode_fuel:.2f} liters")
print(f"Probability of Negative Fuel: {prob_negative_fuel:.4f}")

# 3. Defining a Uniform Prior
prior_min = 0
prior_max = fuel_capacity
prior_pdf = np.where((x >= prior_min) & (x <= prior_max), 1/(prior_max - prior_min), 0)  # Uniform distribution

# 4. Bayesian Update (Grid-Based Approximation)
posterior = pdf_fuel * prior_pdf
posterior /= np.trapz(posterior, x)  # Normalize

plt.figure(figsize=(8,5))
plt.plot(x, pdf_fuel, label="Likelihood (Sensor)", linestyle='dashed')
plt.plot(x, prior_pdf, label="Uniform Prior", linestyle='dotted')
plt.plot(x, posterior, label="Posterior (Updated Estimate)", linewidth=2)
plt.axvline(0, color='r', linestyle='--', label='Zero Fuel')
plt.xlabel("Fuel in Tank (liters)")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Bayesian Update of Fuel Level Estimate (Uniform Prior)")
plt.show()

# Updated probability of negative fuel
prob_negative_fuel_posterior = np.trapz(posterior[x < 0], x[x < 0]) if np.any(x < 0) else 0
print(f"Updated Probability of Negative Fuel: {prob_negative_fuel_posterior:.4f}")

# 5. Bayes Monte Carlo Method
num_samples = 100000
prior_samples = np.random.uniform(prior_min, prior_max, num_samples)  # Uniform prior samples
likelihood_weights = stats.norm.pdf(fuel_sensor_reading, prior_samples, sensor_error_std)
likelihood_weights /= np.sum(likelihood_weights)  # Normalize weights

# Resampling based on posterior weights
posterior_samples = np.random.choice(prior_samples, size=num_samples, p=likelihood_weights)

# Monte Carlo Posterior Histogram
plt.figure(figsize=(8,5))
plt.hist(posterior_samples, bins=50, density=True, alpha=0.6, label="Monte Carlo Posterior")
plt.axvline(0, color='r', linestyle='--', label='Zero Fuel')
plt.xlabel("Fuel in Tank (liters)")
plt.ylabel("Probability Density")
plt.legend()
plt.title("Bayesian Update Using Monte Carlo Method (Uniform Prior)")
plt.show()

# Updated probability of negative fuel using Monte Carlo method
prob_negative_fuel_monte_carlo = np.mean(posterior_samples < 0)
print(f"Monte Carlo Estimated Probability of Negative Fuel: {prob_negative_fuel_monte_carlo:.4f}")

# 7. Estimated Available Flight Time
# Generate random samples for fuel and consumption rate
fuel_samples = np.random.normal(fuel_sensor_reading, sensor_error_std, num_samples)
consumption_samples = np.random.normal(fuel_consumption_rate, consumption_error_std, num_samples)

# Calculate flight time for each sample
flight_times = fuel_samples / consumption_samples

# Plot the distribution of flight times
plt.figure(figsize=(8, 6))
plt.hist(flight_times, bins=100, density=True, alpha=0.6, color='g')
plt.title('Estimated Available Flight Time Distribution')
plt.xlabel('Flight Time (hours)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# (a) Probability that flight time is enough for 100 minutes (1.67 hours) + 0.5 hours (reserve fuel)
min_required_time = 1.67 + 0.5
probability_enough_fuel = np.mean(flight_times >= min_required_time)
print(f"Probability of Having Enough Fuel: {probability_enough_fuel:.4f}")

# (b) Probability that we run out of fuel (i.e., flight time is less than 1.67 hours)
probability_run_out_of_fuel = np.mean(flight_times < 1.67)
print(f"Probability of Running out of Fuel: {probability_run_out_of_fuel:.4f}")
