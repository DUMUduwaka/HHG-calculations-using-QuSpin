from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import numpy as np  
import matplotlib.pyplot as plt


# Define model parameters
L = 100        # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0.15 # Alternating shift of the atos casuing the dimerization (negative for topological phase)

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
        hop_pm_v = hop_pm_v + [[-v, i, (i + 1)]] #(i + 1) % L]
        hop_mp_v = hop_mp_v + [[v, i, (i + 1)  ]]
    else:
        hop_pm_w = hop_pm_w + [[-w, i, (i + 1)  ]]
        hop_mp_w = hop_mp_w + [[w, i, (i + 1) ]]


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

# Plot probability amplitudes against positions of lattice
prob_amplitude = np.transpose(V)**2


Density = np.zeros((L**2,3))
for i in range(L):
    for j in range(L):
        Density[i*L+j,0]= positions[j]
        Density[i*L+j,1]= E[i]
        Density[i*L+j,2]= prob_amplitude[-(i+1),j]


plt.figure(figsize=(8,8))
plt.scatter(Density[:,0], Density[:,1], Density[:,2],marker='.')
plt.ylabel('E(a.u)')
plt.xlabel(r'$X_i(a.u)$')
plt.title(r'$|\Psi_{x}|^{2}$')
plt.savefig(f'Wavefunctions/Density Plot in Real Space δ = {delta}.png')

plt.figure()
plt.plot(positions, prob_amplitude[48,:])
plt.ylabel(r'$|\Psi_{48}|^{2}$')
plt.title(r'$|\Psi_{48}|^{2}$')
plt.savefig(f'Wavefunctions/Psi48 for δ = {delta}.png')

plt.figure()
plt.plot(positions, prob_amplitude[49,:])
plt.ylabel(r'$|\Psi_{49}|^{2}$')
plt.title(r'$|\Psi_{49}|^{2}$')
plt.savefig(f'Wavefunctions/Psi49 for δ = {delta}.png')

plt.figure()
plt.plot(positions, prob_amplitude[50,:])
plt.ylabel(r'$|\Psi_{50}|^{2}$')
plt.title(r'$|\Psi_{50}|^{2}$')
plt.savefig(f'Wavefunctions/Psi50 for δ = {delta}.png')

plt.figure()
plt.plot(positions, prob_amplitude[51,:])
plt.ylabel(r'$|\Psi_{51}|^{2}$')
plt.title(r'$|\Psi_{51}|^{2}$')
plt.savefig(f'Wavefunctions/Psi51 for δ = {delta}.png')



# Ploting the wavefunction in k-space
x = L*a
N = L
delta_x = x/N # positions[1]-positions[0]


k = np.linspace(-1*np.pi/delta_x, np.pi/delta_x, N, endpoint= False)
delta_k = k[1]-k[0]
print(k)

factor = np.zeros(N)
for i in range(N):
    factor[i] = np.exp(1j*k[0]*i*delta_x)

V_k = np.zeros((L,L))
for i in range(L):
    V[:,i] = factor*V[:,i]
    V_k[:,i] = (delta_x/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(V[:,i])

prob_amplitude_k= np.absolute(np.transpose(V_k)**2)


Density_k = np.zeros((L**2,3))
for i in range(L):
    for j in range(L):
        Density_k[i*L+j,0]= k[j]
        Density_k[i*L+j,1]= E[i]
        if prob_amplitude_k[i,j] < 1e-2 and -1e-10<E[i]<1e-10:
            Density_k[i*L+j,2]= 1e-30
        else:
            Density_k[i*L+j,2]= prob_amplitude_k[-(i+1),j]

plt.figure(figsize=(8,8))
plt.scatter(Density_k[:,0], Density_k[:,1], Density_k[:,2],marker='o',linewidth=0)
plt.ylabel('E(a.u)')
plt.xlabel('k(a.u)')
plt.title(r'$|\Psi_{x}|^{2}$')
plt.savefig(f'Wavefunctions/Density Plot in k-space δ = {delta}.png')