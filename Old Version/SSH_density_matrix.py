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
import matplotlib.cm as cm
import os

#######################################
######  Density Matix  Method   #######
#######################################

# Create output directory if not exists
output_dir = "Density Matrix"
os.makedirs(output_dir, exist_ok=True)


# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0.15      # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
N_cyc = 5                            # Period of the pulse
omega_0 = 0.0075                     # Frequency in a.u
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
time_step = 0.1
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

# Solving the time-dependent Hamiltonian to obatin L/2 number of eigenstates correspond to lowest eigenvalues
E , V = H_t.eigsh(time=0.0, k=L/2,which="SA")

V_evolved = H_t.evolve(V,0,t,eom='SE',iterate=True)
V_evolved_list = list(V_evolved)


displacement_total=np.zeros(len(t))
current_total = np.zeros(len(t))

for time in range(len(t)):
    state = V_evolved_list[time]
    Rho = np.zeros((L,L), dtype=complex)

    for i in range (L//2):
        mat_1 = state[:,i].reshape(-1,1)
        mat_2 = np.conjugate(state[:,i]).reshape(1,-1)
        Rho += np.matmul(mat_1,mat_2)
        
    mat_dis = displacement.toarray(time=t[time])
    mat_current = current.toarray(time=t[time])

    res_dis = np.matmul(Rho,mat_dis)
    res_current = np.matmul(Rho,mat_current)
    
    displacement_total[time]= np.trace(res_dis)
    current_total[time]= np.trace(res_current)



# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]

factor = np.zeros(num)
for k in range(num):
    factor[k] = np.exp(1j*omega[0]*k*delta_t)


# Define the mask
Mask = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)


#print("Current using the density matrix ", current_time_den_mat)
plt.figure()
plt.plot(t*t_conversion,displacement_total*Mask, linewidth=1.0)
plt.xlabel('Time (fs)')
plt.ylabel('Displacement')
plt.title("Displacement vs Time using density matrix")
plt.savefig(f'Density Matrix/Displacement using density matrix δ = {delta}, Δt = {time_step}.png')


displacement_total= factor*displacement_total*Mask
X_omega_den_mat = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total)
P_omega_den = abs(omega**2*X_omega_den_mat)**2

current_total= factor*current_total*Mask
J_omega_den_mat = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(current_total)
S_omega_den = abs(omega*J_omega_den_mat)**2

omega_new = omega/omega_0

## Plot High Harmonic Spectra
plt.figure(figsize=(8, 6))
plt.plot(omega_new,P_omega_den,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.title(fr'P($\omega$) using the density matrix for displacement δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'Density Matrix/P(Omega) using density matrix for displacement δ = {delta}, Δt = {time_step}.png')

plt.figure(figsize=(8, 6))
plt.plot(omega_new,S_omega_den,linewidth=1.0)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.title(fr'P($\omega$) using the density matrix for current δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'Density Matrix/P(Omega) using density matrix for current δ = {delta}, Δt = {time_step}.png')

plt.figure()
plt.plot(omega_new, S_omega_den, label="Current", linewidth=1.0)
plt.plot(omega_new, P_omega_den, label="Displacement", linewidth=1.0)
plt.legend()
plt.yscale('log')
plt.ylabel(r'$S(\omega)$ using density matrix method')
plt.xlim(0, 100)
plt.xlabel(r'$\omega$')
plt.tight_layout()  # Optional: to prevent cutoff
plt.savefig(f'Density Matrix/S(Omega) Comparison δ = {delta}, Δt = {time_step}.png')