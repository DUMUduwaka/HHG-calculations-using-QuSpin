from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time
import scipy as sp
import numpy as np  # generic math functions
import matplotlib.pyplot as plt


# Define model parameters
L = 100               # system size
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
def current_ramp(t,F_0,omega_0,NN,a):
    A_t = (F_0*10/omega_0)*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)
    return -1j*a*np.exp(-1j*A_t)

def current_ramp_conj(t,F_0,omega_0,NN,a):
    A_t = (F_0*10/omega_0)*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)
    return 1j*a*np.exp(1j*A_t)

current_ramp_args = [F_0,omega_0,NN,a]

current_static = []
current_dynamic = [["+-", hop_pm,current_ramp,current_ramp_args], ["-+", hop_mp,current_ramp_conj,current_ramp_args]]
current = hamiltonian(current_static,current_dynamic, basis=basis,dtype=np.float64)

# calculate expectation values of current operator
Obs_time = obs_vs_time(psi_t,t,dict(Energy=H_t, Current=current))
current_time = Obs_time["Current"]
print(np.size(current_time))

plt.plot(t,current_time)
#plt.show()



current_omega = sp.fft.fft(current_time)

print(current_omega)
plt.plot(t,current_time)
#plt.show()

delta_t = t[1]-t[0]
omega = sp.fft.fftfreq(num, delta_t) * 2 * np.pi  # Frequency array in rad/s

# Obtain omega * J(omega)
omega_J_omega = omega * current_omega
S_omega = abs(omega_J_omega)**2
# Plot J(t) and omega * J(omega) (magnitude)
plt.figure(figsize=(14, 6))

# Plot J(t)

plt.plot(S_omega)
plt.show()







