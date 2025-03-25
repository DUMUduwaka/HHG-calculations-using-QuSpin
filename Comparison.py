from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt
import csv
import os


# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0         # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
N_cyc = 5                            # Period of the pulse
omega_0 = 0.0075                     # Frequency in a.u
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

displacement_total_CN = np.zeros(len(t))

for time in range(len(t)):
    if time==0:
        state = V
    else:
        state = evolve_state(t[time],state)

    x_t = 0
    for i in range(L//2):
        for j in range(0, L):
            x_t += np.conjugate(state[j,i])*positions[j]*state[j,i]
            
    displacement_total_CN[time] = x_t


#################################
######  QuSpin Operators  #######
#################################

displacement_total_QS = np.zeros(len(t))

for s in range(L//2):
    psi_0 = V[:,s]

    # Evolve State
    psi_t = H_t.evolve(psi_0,0,t, eom='SE', iterate= True)
    Obs_time = obs_vs_time(psi_t,t,dict( Displacement=displacement))
    
    displacement_total_QS += np.real(Obs_time["Displacement"])


###############################
######  Density Matrix  #######
###############################

V_evolved = H_t.evolve(V,0,t,eom='SE',iterate=True)
V_evolved_list = list(V_evolved)

displacement_total_DM = np.zeros(len(t))
for i in range(L//2):
     
    displacement_time = np.zeros(len(t))
    for time in range(len(t)):
        state = V_evolved_list[time]
        #Rho = np.zeros((L,L))

        mat_1 = state[:,i].reshape(-1,1)
        mat_2 = np.conjugate(state[:,i]).reshape(1,-1)
        Rho = np.matmul(mat_1,mat_2)

        mat_3 = displacement.toarray(time=t[time])

        res = np.matmul(Rho, mat_3)

        displacement_time[time]=np.trace(res)
        
    displacement_total_DM += displacement_time


# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]
omega_new = omega/omega_0

# Define the mask
Mask = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)


factor = np.zeros(num)
for k in range(num):
    factor[k] = np.exp(1j*omega[0]*k*delta_t)


# Fourier Transforamtion of dispalcement and acceleration
displacement_total_CN = factor*displacement_total_CN*Mask
X_omega_CN = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total_CN)
P_omega_CN = abs(omega**2*X_omega_CN)**2

displacement_total_QS = factor*displacement_total_QS*Mask
X_omega_QS = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total_QS)
P_omega_QS = abs(omega**2*X_omega_QS)**2

displacement_total_DM = factor*displacement_total_DM*Mask
X_omega_DM = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total_DM)
P_omega_DM = abs(omega**2*X_omega_DM)**2

# Plot P_omega
plt.figure(figsize=(10, 8))
plt.plot(omega_new,P_omega_CN,linewidth=1.0, color='r', label = 'Crank-Nicolson')
plt.plot(omega_new,P_omega_QS,linewidth=1.0, color='k', label ='QuSpin Operators')
plt.plot(omega_new,P_omega_DM,linewidth=1.0, color='g', label = 'Density Matrix')
plt.yscale('log')
plt.ylabel(r'P($\omega$) using displacement')
plt.title(f'P($\Omega$) Comparison δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.legend(loc='best')
plt.savefig(f'P(Omega) Comparison for different methods δ = {delta}, Δt = {time_step}.png')


# Subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))
fig.suptitle(f'P($\Omega$) Comparison δ = {delta}, Δt = {time_step}', fontsize=16)

# Plot for Crank-Nicolson method
ax1.plot(omega_new, P_omega_CN, linewidth=1.0, color='r')
ax1.set_yscale('log')
ax1.set_ylabel(r'P($\omega$) using displacement')
ax1.set_title('Crank-Nicolson')
ax1.set_xlim(0, 100)

# Plot for QuSpin Operators method
ax2.plot(omega_new, P_omega_QS, linewidth=1.0, color='k')
ax2.set_yscale('log')
ax2.set_ylabel(r'P($\omega$) using displacement')
ax2.set_title('QuSpin Operators')
ax2.set_xlim(0, 100)

# Plot for Density Matrix method
ax3.plot(omega_new, P_omega_DM, linewidth=1.0, color='g')
ax3.set_yscale('log')
ax3.set_ylabel(r'P($\omega$) using displacement')
ax3.set_title('Density Matrix')
ax3.set_xlim(0, 100)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig(f'P(Omega) Comparison for different methods δ = {delta}, Δt = {time_step} subplots.png')