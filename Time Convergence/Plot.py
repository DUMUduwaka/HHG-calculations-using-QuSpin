import matplotlib.pyplot as plt
import csv
from collections import defaultdict

delta = 0.15
# Read data from the CSV
data = defaultdict(list)
input_file = f"Time Convergence/S_omega_data_δ={delta}.csv"

with open(input_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        time_step = float(row["Time_step"])
        omega_new = float(row["Omega_New"])
        s_omega = float(row["S_Omega"])
        data[time_step].append((omega_new, s_omega))

# Plot the data
plt.figure(figsize=(10, 8))
for time_step, values in data.items():
    omega_new, s_omega = zip(*values)
    plt.plot(omega_new, s_omega, label=f"Δt: {time_step}")

plt.xlabel("Normalized Frequency (ω/ω₀)")
plt.ylabel(r'S($\omega$)')
plt.yscale('log')
plt.xlim(0,60)
plt.title(r'S($\omega$) for Different Time Steps')
plt.legend()
plt.grid(True)
plt.savefig(f'Time Convergence/TC δ = {delta}.png')


# Create subplots (3 columns, 2 rows)
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

# Flatten axes array for easier indexing
axs = axs.flatten()

# Plot for each time step
for idx, (time_step, values) in enumerate(sorted(data.items())):
    omega_new, s_omega = zip(*values)
    axs[idx].plot(omega_new, s_omega, label=f"Δt: {time_step}", color='teal')
    axs[idx].set_title(f"Δt: {time_step}")
    axs[idx].set_ylabel(r'S($\omega$)')
    axs[idx].set_yscale('log')
    axs[idx].set_xlim(0, 60)
    axs[idx].grid(True)
    axs[idx].legend()

# Hide unused subplots if there are fewer than 6 time steps
for ax in axs[len(data):]:
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.savefig(f'Time Convergence/TC_subplots δ = {delta}.png')

