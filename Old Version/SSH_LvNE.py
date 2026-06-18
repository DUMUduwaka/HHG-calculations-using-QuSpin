'''
Author: Don Usitha Mihiranga Uduwakaarachchi
Copyright (c) 2025 Don Usitha Mihiranga Uduwakaarachchi
'''

import os
from quspin.operators import hamiltonian, commutator, anti_commutator
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis_1d
from quspin.tools.evolution import evolve
from quspin.tools.misc import get_matvec_function
import numpy as np
import scipy as sp
from six import iteritems  # loop over elements of dictionary
import matplotlib.pyplot as plt  # plotting library

####################################
######  LiNE equation Method  ######
####################################

# Create output directory if not exists
output_dir = f"LiNE EOM"
os.makedirs(output_dir, exist_ok=True)

# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = -0.15    # Alternating shift of the atos casuing the dimerization (negative for topological phase)


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


# Define site-coupling lists with out boundry conditions
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
#H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)


# define the Hamitonian in the presence of the external field
stat = []
dyna = [["+-", hop_pm_v,ramp_v,ramp_args], 
        ["+-", hop_pm_w,ramp_w,ramp_args],
        ["-+", hop_mp_v,ramp_v_conj,ramp_args],
        ["-+", hop_mp_w,ramp_w_conj,ramp_args]]

H = hamiltonian(stat,dyna,basis=basis, dtype=np.float64)

# Define current operator
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


##### define LiNE equation in diagonal form
def LiNE_EOM(time, rho):
    rho = rho.reshape((L,L))
    rho_dot = -1j * (np.matmul(H.toarray(time=time),rho)-np.matmul(rho,H.toarray(time=time)))
 
    return rho_dot.ravel()

# Solving the time-dependent Hamiltonian to obatin L/2 number of eigenstates correspond to lowest eigenvalues
E , V = H.eigsh(time=0.0, k=L/2,which="SA")

#Initial Density Matirx
Rho = np.zeros((L,L),dtype=complex)
for i in range (L//2):
    mat_1 = V[:,i].reshape(-1,1)
    mat_2 = np.conjugate(V[:,i]).reshape(1,-1)
    Rho += np.dot(mat_1,mat_2)
rho0 = Rho

#Evolve density Matrix
rho_t = evolve(rho0,t[0],t,LiNE_EOM,iterate=True,atol=1E-12,rtol=1E-12)


displacement_time = np.zeros(len(t))
current_time = np.zeros(len(t))
for i, rho_flattened in enumerate(rho_t):
    rho = rho_flattened.reshape(H.Ns, H.Ns)
    mat_dis = displacement.toarray(time=t[i])
    mat_current = current.toarray(time=t[i])
    
    res_dis = np.matmul(rho,mat_dis)
    res_current = np.matmul(rho,mat_current)

    displacement_time[i]= np.trace(res_dis)
    current_time[i]= np.trace(res_current)


# Define the mask
Mask = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)


plt.figure()
plt.plot(t*t_conversion,displacement_time*Mask)
plt.xlabel('Time (fs)')
plt.ylabel('Displacement')
plt.title("Displacement vs Time")
plt.savefig(f'LiNE EOM/Displacement δ = {delta}.png')

# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]

factor = np.zeros(num)
for k in range(num):
    factor[k] = np.exp(1j*omega[0]*k*delta_t)


omega_new = omega/omega_0

displacement_total = factor*displacement_time*Mask
X_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total)
P_omega = abs(omega**2*X_omega)**2

current_total = factor*current_time*Mask
J_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(current_total)
S_omega = abs(omega*J_omega)**2

# Plot P_omega
plt.figure(figsize=(8, 6))
plt.plot(omega_new,P_omega,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f'P($\omega$) using displacement (LiNE method), δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'LiNE EOM/P(Omega) using LiNE eom and displacement δ = {delta}, Δt = {time_step}.png')

# Plot P_omega
plt.figure(figsize=(8, 6))
plt.plot(omega_new,S_omega,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f'P($\omega$) using current (LiNE method), δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'LiNE EOM/P(Omega) using LiNE eom and current δ = {delta}, Δt = {time_step}.png')


plt.figure(figsize=(8, 6))
plt.plot(omega_new, S_omega, label="Current", linewidth=1.0)
plt.plot(omega_new, P_omega, label="Displacement", linewidth=1.0)
plt.legend()
plt.yscale('log')
plt.ylabel(r'$S(\omega)$ using density matrix LvNE method')
plt.xlim(0, 100)
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f'P($\omega$) comparison (LiNE method), δ = {delta}, Δt = {time_step}')
plt.tight_layout()  # Optional: to prevent cutoff
plt.savefig(f'LiNE EOM/S(Omega) Comparison δ = {delta}, Δt = {time_step}.png')