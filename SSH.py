from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools import measurements #
import numpy as np  # generic math functions
import matplotlib.pyplot as plt


# Define model parameters
L = 10               # system size
J = 1.0              # uniform hopping contribution
deltaJ = 0.1         # bond dimerisation
Delta = 0.5          # staggered potential

# Declare constants for Vector Potential
a = 4                # Lattice constant in Å
NN = 10              # Period of the pulse
omega_0 = 32.9       # Frequency in THz
F_0 = 10             # Amplitude in MV/cm
t_0 = 0.52           # Intra chain hopping parameter in eV


# Define the hopping elements 
v = J + deltaJ # intracell hopping parameter
w = J - deltaJ # intercell hopping parameter

# Define site-coupling lists
hop_pm = []
hop_mp = []

for i in range(L):
    if i%2 == 0:
        hop_pm = hop_pm + [[-v, i, (i + 1) % L]]
        hop_mp = hop_mp + [[v, i, (i + 1) % L]]
    else:
        hop_pm = hop_pm + [[-w, i, (i + 1) % L]]
        hop_mp = hop_mp + [[w, i, (i + 1) % L]]

'''
# hop_pm = [[-J - deltaJ * (-1) ** i, i, (i + 1) % L] for i in range(L)]  # PBC
# hop_mp = [[+J + deltaJ * (-1) ** i, i, (i + 1) % L] for i in range(L)]  # PBC
'''

start,stop,num = 0, 2, 100     # time in fs
t = np.linspace(start, stop, num=num)      # Time array
A_t = (F_0*10/omega_0)*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)     # Vector Potential

# Plot Vector potential; A(t)
plt.plot(t,A_t)
plt.xlabel("t(fs)")
plt.ylabel("A(t)")
#plt.show()

def ramp(t,F_0,omega_0,NN,a):
    A_t = (F_0*10/omega_0)*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)
    return np.exp(-1j*A_t)

def ramp_conj(t,F_0,omega_0,NN,a):
    A_t = (F_0*10/omega_0)*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)
    return np.exp(1j*A_t)

ramp_args = [F_0,omega_0,NN,a]

'''
print(ramp_args)
plt.plot(t,ramp(t,F_0,omega_0,NN,a))
plt.show()
'''

## Construct single-particle Hamiltonian 

# define basis
basis = spinless_fermion_basis_1d(L, Nf=1)

# define static and dynamic lists and build real-space Hamiltonian
static = [["+-", hop_pm], ["-+", hop_mp]]
dynamic = []
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)


# define the Hamitonian in the presence of the external field
stat = []
dyna = [["+-", hop_pm,ramp,ramp_args], ["-+", hop_mp,ramp_conj,ramp_args]]
H_t = hamiltonian(stat,dyna,basis=basis, dtype=np.float64)

# Define the initial state
E_0 , psi_0 = H_t.eigh(time=0.0)
psi_0 = psi_0[:,0]
print(psi_0)

# Evolve state in 
psi_t = H_t.evolve(psi_0,0,t,eom='SE',iterate=True)
print(psi_t)



# Define current operator
current_static = []
current_dynamic = []













