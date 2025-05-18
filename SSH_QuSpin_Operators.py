'''
Author: Don Usitha Mihiranga Uduwakaarachchi
Copyright (c) 2025 Don Usitha Mihiranga Uduwakaarachchi
'''

from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt
import os

#################################
######  QuSpin Operators  #######
#################################

# Create output directory if not exists
output_dir = f"QuSpin Operators"
os.makedirs(output_dir, exist_ok=True)

# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0.15      # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
N_cyc = 5                            # Period of the pulse
omega_0 = 0.0075                      # Frequency in a.u
A_0 = 0.2                            # Amplitude 
tf = 2*np.pi*N_cyc/omega_0           # Final time
t_conversion = 2.4188843265864e-2    # Conversion of time from a.u to fs


# Define atomic-site positions
positions = np.zeros(L)
for i in range(L):
    positions[i]=((i+1)-((L+1)/2))*a-((-1)**(i+1))*delta


# Define the hopping elements using the distances between atoms
v = -np.exp(-(a-2*delta)) # intracell hopping parameter
w = -np.exp(-(a+2*delta)) # intercell hopping parameter


# Define site-coupling lists without boundary conditions
hop_pm_v = []
hop_pm_w = []
hop_mp_v = []
hop_mp_w = []

for i in range(L-1):
    if i%2 == 0:
        hop_pm_v = hop_pm_v + [[-v, i, i+1]]
        hop_mp_v = hop_mp_v + [[v, i, i+1]]
    else:
        hop_pm_w = hop_pm_w + [[-w, i, i+1]]
        hop_mp_w = hop_mp_w + [[w, i, i+1]]


# Define the time array and the Vector potential 
time_step = 1 
start,stop,num = 0, tf, int(tf/time_step)                        # time in fs
t = np.linspace(start, stop, num=num, endpoint=False)            # Time array
A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)   # Vector Potential


# Define time dependent parts in the Hamiltonian
def ramp_v(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return np.exp((-1j*(a-delta)*A_t))

def ramp_w(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return np.exp((-1j*(a+delta)*A_t))

def ramp_v_conj(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return np.exp(1j*(a-delta)*A_t)

def ramp_w_conj(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return np.exp(1j*(a+delta)*A_t)

ramp_args = [A_0,omega_0,N_cyc,a,delta]


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


# define the Hamitonian in the presence of the external field
stat = []
dyna = [["+-", hop_pm_v,ramp_v,ramp_args], 
        ["+-", hop_pm_w,ramp_w,ramp_args],
        ["-+", hop_mp_v,ramp_v_conj,ramp_args],
        ["-+", hop_mp_w,ramp_w_conj,ramp_args]]

H_t = hamiltonian(stat,dyna,basis=basis, dtype=np.float64)

## Define current operator
# Define time dependent parts in the current
def current_ramp_v(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return 1j*(a-2*delta)*np.exp((-1j*(a-delta)*A_t))

def current_ramp_w(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return 1j*(a+2*delta)*np.exp((-1j*(a+delta)*A_t))

def current_ramp_v_conj(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return -1j*(a-2*delta)*np.exp(1j*(a-delta)*A_t)

def current_ramp_w_conj(t,A_0,omega_0,N_cyc,a, delta):
    A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t) 
    return -1j*(a+2*delta)*np.exp(1j*(a+delta)*A_t)

current_ramp_args = [A_0,omega_0,N_cyc,a,delta]


# define the Hamitonian in the presence of the external field
current_static = []
current_dynamic = [["+-", hop_pm_v,current_ramp_v,current_ramp_args], 
                   ["+-", hop_pm_w,current_ramp_w,current_ramp_args],
                   ["-+", hop_mp_v,current_ramp_v_conj,current_ramp_args],
                   ["-+", hop_mp_w,current_ramp_w_conj,current_ramp_args]]

current = hamiltonian(current_static,current_dynamic, basis=basis,dtype=np.float64)


# Define position operator
X_i = [[positions[i],i] for i in range(L)]

displacement_static=[["n",X_i]]
displacement_dynamic=[]
displacement = hamiltonian(displacement_static,displacement_dynamic,basis=basis, dtype=np.float64)

# Solving the time-dependent Hamiltonian to obatin L/2 number of eigenstates correspond to lowest eigenvalues
E , V = H_t.eigsh(time=0.0, k=L/2,which="SA")

# Store eigenstates in a list
eigenstates = [V[:,i] for i in range(int(L/2))]

current_total = np.zeros(len(t),dtype=complex)
displacement_total = np.zeros(len(t),dtype=complex)

for i, psi_0 in enumerate(eigenstates):
    # Evolve state 
    psi_t = H_t.evolve(psi_0,0,t,eom='SE',iterate=True)
   
    Obs_time = obs_vs_time(psi_t,t,dict(Energy=H_t,Current=current, Displacement=displacement))
    current_time = Obs_time["Current"]              # Expectation Value of Current
    displacement_time = Obs_time["Displacement"]    # Expectation Value of Displacement
   
    current_total += current_time
    displacement_total += displacement_time


# Define the mask
Mask = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)

# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]

factor = np.zeros(num)
for k in range(num):
    factor[k] = np.exp(1j*omega[0]*k*delta_t)

current_total = factor*current_total*Mask
J_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(current_total)
omega_J_omega = omega * J_omega
S_omega = abs(omega_J_omega)**2

displacement_total = factor*displacement_total*Mask
X_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total)
P_omega = abs(omega**2*X_omega)**2

omega_new = omega/omega_0

## Plot High Harmonic Spectra
plt.figure(figsize=(8, 6))
plt.plot(omega_new,P_omega,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f'P($\omega$) using displacement operator δ = {delta}')
plt.xlim(0,100)
plt.savefig(f'QuSpin Operators/P(omega) using displacement δ = {delta}, Δt = {time_step}.png')

plt.figure(figsize=(8, 6))
plt.plot(omega_new,S_omega,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f'P($\omega$) using current operator δ = {delta}')
plt.xlim(0,100)
plt.savefig(f'QuSpin Operators/P(omega) using current δ = {delta}, Δt = {time_step}.png')

plt.figure(figsize=(8, 6))
plt.plot(omega_new, S_omega, label="Current", linewidth=1.0)
plt.plot(omega_new, P_omega, label="Displacement", linewidth=1.0)
plt.legend()
plt.yscale('log')
plt.ylabel(r'$P(\omega)$')
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f'P($\omega$) comparison δ = {delta}, Δt = {time_step}')
plt.xlim(0, 100)
plt.tight_layout()  # Optional: to prevent cutoff
plt.savefig(f'QuSpin Operators/S(Omega) Comparison δ = {delta}, Δt = {time_step}.png')