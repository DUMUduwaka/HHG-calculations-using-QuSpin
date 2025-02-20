import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = -0.15    # Alternating shift of the atos casuing the dimerization (negative for topological phase)

# Define atomic-site positions
positions = np.zeros(L)
for i in range(L):
    positions[i]=((i+1)-((L+1)/2))*a-((-1)**(i+1))*delta

# Plot the chain positions
plt.figure(figsize=(15, 3))
plt.scatter(positions, np.zeros(L), color='blue', label='Atomic Sites')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Add reference line
plt.xlabel("Position (a.u.)")
plt.title("Atomic Positions in the SSH Chain")
plt.yticks([])  # Remove y-axis ticks
plt.legend()
plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(f'Atomic Positions/Atomic Postions in the SSH chain δ = {delta}.png')