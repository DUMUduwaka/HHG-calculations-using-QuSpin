from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np

# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = -0.15     # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
NN = 5                               # Period of the pulse
omega_0 = 0.0075                     # Frequency in THz
A_0 = 0.2                            # Amplitude 
tf = 2*np.pi*NN/omega_0              # Final time
t_conversion = 2.4188843265864e-2    # Conversion of time from a.u to fs


# Define atomic-site positions
positions = np.zeros(L)
for i in range(L):
    positions[i]=((i+1)-((L+1)/2))*a-((-1)**(i+1))*delta


# Define the hopping elements using the distances between atoms
v = -np.exp(-(a-2*delta)) # intracell hopping parameter
w = -np.exp(-(a+2*delta)) # intercell hopping parameter


# Define site-coupling lists with out boundry conditions
hop_pm_v = []
hop_pm_w = []
hop_mp_v = []
hop_mp_w = []

for i in range(L-1):
    if i%2 == 0:
        hop_pm_v = hop_pm_v + [[-v, i, i+1]] #(i + 1) % L]
        hop_mp_v = hop_mp_v + [[v, i, i+1]]
    else:
        hop_pm_w = hop_pm_w + [[-w, i, i+1]]
        hop_mp_w = hop_mp_w + [[w, i, i+1]]



# Define the time array and the Vector potential 
start,stop,num = 0, tf, 200                                   # time in fs
t = np.linspace(start, stop, num=num, endpoint=False)         # Time array
A_t = A_0*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)   # Vector Potential



## Construct single -praticle Hamiltonian

# define basis
basis = spinless_fermion_basis_1d(L, Nf=1)

# define static and dynamic lists and build real-space Hamiltonian
static = [["+-", hop_pm_v],
          ["+-", hop_pm_w], 
          ["-+", hop_mp_v],
          ["-+", hop_mp_w]]
dynamic = []
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
E,V = H.eigh() # the Smallest Algeberic Eigen value and conresponding eigen state. 

prob_amplitude = np.transpose(V)**2

print("Energy is", E)

dense = np.zeros((L*L,3))

for i in range(len(E)):
    for j in range(len(positions)):
        k = i*len(positions)+j
        dense[k,0] = positions[j]
        dense[k,1] = E[i]
        dense[k,2] = prob_amplitude[i,j]
print(dense)

x, y, z = dense[:,0],dense[:,1],dense[:,1]


# Plot probability amplitudes against positions of lattice
x_positions = positions
y_positions = E
prob_amplitude = np.transpose(V)**2

X,Y = np.meshgrid(x_positions,y_positions)

density = np.zeros((len(x_positions),len(y_positions)))
for i in range(len(x_positions)):
    for j in range(len(y_positions)):
        density[i,j]= prob_amplitude[i,j]


plt.figure()
plt.plot(x_positions, prob_amplitude[48,:])
plt.ylabel(r'$|\Psi_{48}|^{2}$')
plt.title(r'$|\Psi_{48}|^{2}$')
plt.savefig('Plots/Psi48.png')

plt.figure()
plt.plot(x_positions, prob_amplitude[49,:])
plt.ylabel(r'$|\Psi_{49}|^{2}$')
plt.title(r'$|\Psi_{49}|^{2}$')
plt.savefig('Plots/Psi49.png')

plt.figure()
plt.plot(x_positions, prob_amplitude[50,:])
plt.ylabel(r'$|\Psi_{50}|^{2}$')
plt.title(r'$|\Psi_{50}|^{2}$')
plt.savefig('Plots/Psi50.png')

plt.figure()
plt.plot(x_positions, prob_amplitude[51,:])
plt.ylabel(r'$|\Psi_{51}|^{2}$')
plt.title(r'$|\Psi_{51}|^{2}$')
plt.savefig('Plots/Psi51.png')