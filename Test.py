from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt
import csv
import os


# Define model parameters
L = 100          # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = -0.15     # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
N_cyc = 5                            # Period of the pulse
omega_0 = 0.0075                      # Frequency in a.u
A_0 = 0.2                            # Amplitude 
tf = 2*np.pi*N_cyc/omega_0           # Final time
t_conversion = 2.4188843265864e-2    # Conversion of time from a.u to fs

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


## Construct single-praticle Hamiltonian
# define basis
basis = spinless_fermion_basis_1d(L, Nf=1)

# define the Hamitonian in the presence of the external field
stat = []
dyna = [["+-", hop_pm_v,ramp_v,ramp_args], 
        ["+-", hop_pm_w,ramp_w,ramp_args],
        ["-+", hop_mp_v,ramp_v_conj,ramp_args],
        ["-+", hop_mp_w,ramp_w_conj,ramp_args]]


# Define position operator

X_i = [[positions[i],i] for i in range(L)]

displacement_static=[["n",X_i]]
displacement_dynamic=[]
displacement = hamiltonian(displacement_static,displacement_dynamic,basis=basis, dtype=np.float64)

H_t = hamiltonian(stat,dyna,basis=basis, dtype=np.float64)
E , V = H_t.eigsh(time=0.0, k=L/2,which="SA")

delta_t = t[1]-t[0]   # Time step
I = np.eye(L)


###########################################
######  Crank-Nicolson Propagation  #######
###########################################


def evolve_state(t,state):
    H = H_t.toarray(time=t+delta_t/2)
    mat_a = I+1j*H*delta_t/2
    return np.linalg.solve(mat_a, state-np.dot(1j*H*delta_t/2,state))

displacement_time = np.zeros(len(t))

for time in range(len(t)):
    if time==0:
        state = V
    else:
        state = evolve_state(t[time],state)

    x_t = 0
    for i in range(L//2):
        for j in range(0, L):
            x_t += np.conjugate(state[j,i])*positions[j]*state[j,i]
            
    displacement_time[time] = x_t


# Define the mask
Mask = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)


plt.figure()
plt.plot(t*t_conversion,displacement_time*Mask)
plt.xlabel('Time (fs)')
plt.ylabel('Displacement')
plt.title("Displacement vs Time")
plt.savefig(f'Displacement δ = {delta}.png')

# Using QuSpin

# Store eigenstates in a list
'''
eigenstates = [V[:,i] for i in range(SS,SS+10)]
displacement_total = np.zeros(len(t))


for i, psi_0 in enumerate(eigenstates):
    # Evolve state 
    psi_t = H_t.evolve(psi_0,0,t,eom='SE',iterate=True)
   
    Obs_time = obs_vs_time(psi_t,t,dict( Displacement=displacement))

    displacement_time_Qu = np.real(Obs_time["Displacement"])
    displacement_total =+ displacement_time_Qu 

'''
displacement_total = np.zeros(len(t))

for s in range(L//2):
    psi_0 = V[:,s]

    # Evolve State
    psi_t = H_t.evolve(psi_0,0,t, eom='SE', iterate= True)
    Obs_time = obs_vs_time(psi_t,t,dict( Displacement=displacement))
    
    displacement_total += np.real(Obs_time["Displacement"])

# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]
omega_new = omega/omega_0


factor = np.zeros(num)
for k in range(num):
    factor[k] = np.exp(1j*omega[0]*k*delta_t)


# Fourier Transforamtion of dispalcement and acceleration
displacement_time = factor*displacement_time*Mask
X_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_time)
P_omega = abs(omega**2*X_omega)**2


displacement_total = factor*displacement_total*Mask
X2_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total)
P2_omega = abs(omega**2*X2_omega)**2


# Plot P_omega
plt.figure(figsize=(8, 6))
plt.plot(omega_new,P_omega,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$) using displacement')
plt.title(f'P($\Omega$) using Displacement δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'P(Omega) using Displacement δ = {delta}, Δt = {time_step}, state={SS}.png')

'''
# Plot P2_omega
plt.figure(figsize=(8, 6))
plt.plot(omega_new,P2_omega,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P2($\omega$) using displacement')
plt.title(f'P2($\Omega$) using Displacement δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'P2(Omega) using Displacement δ = {delta}, Δt = {time_step}.png')
'''