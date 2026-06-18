import os
from quspin.operators import hamiltonian, commutator, anti_commutator
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spin basis_1d
from quspin.tools.evolution import evolve
from quspin.tools.misc import get_matvec_function
import numpy as np
import scipy as sp
from six import iteritems  # loop over elements of dictionary
import matplotlib.pyplot as plt  # plotting library

# Create output directory if not exists
output_dir = f"test"
os.makedirs(output_dir, exist_ok=True)

# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0.15    # Alternating shift of the atos casuing the dimerization (negative for topological phase)


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

for i in range(L):
    if i%2 == 0:
        hop_pm_v = hop_pm_v + [[-v, i, (i+1)%L]]
        hop_mp_v = hop_mp_v + [[v, i,(i+1)%L]]
    else:
        hop_pm_w = hop_pm_w + [[-w, i,(i+1)%L]]
        hop_mp_w = hop_mp_w + [[w, i, (i+1)%L]]


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


# Define position operator

X_i = [[positions[i],i] for i in range(L)]

displacement_static=[["n",X_i]]
displacement_dynamic=[]
displacement = hamiltonian(displacement_static,displacement_dynamic,basis=basis, dtype=np.float64)


#Dephasing time (T2)
#T2_fs = 10
#T2 = T2_fs/t_conversion  # Convert fs to a.u
#print(T2)

Energies , U = H.eigh(time=0.0)
##### define LiNE equation in diagonal form including Dephasing

def Dephase(rho, T2, U):

    new_rho = np.copy(rho)
    new_rho = np.matmul(np.transpose(np.conjugate(U)),np.matmul(new_rho,U))
    new_diagonal = np.zeros(new_rho.shape[0], dtype=complex)
    np.fill_diagonal(new_rho, new_diagonal)
    new_rho = np.matmul(U,np.matmul(new_rho,np.transpose(np.conjugate(U))))
    return new_rho/T2

T2_val = [5, 10 , np.inf]
plt.figure(figsize=(8,6))
for i in range(len(T2_val)):
    T2_fs= T2_val[i]
    T2 = T2_fs/t_conversion

    def LiNE_EOM(time, rho):
        rho = rho.reshape((L,L))
        rho_dot = -1j * (np.matmul(H.toarray(time=time),rho)-np.matmul(rho,H.toarray(time=time)))- Dephase(rho,T2,U)

        return rho_dot.ravel()


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

    """
    population_time = np.zeros(len(t))
    for i, rho_flattened in enumerate(rho_t):
        rho = rho_flattened.reshape(H.Ns, H.Ns)
        population_time[i]= np.trace(rho)

    plt.figure()
    plt.plot(t*t_conversion,population_time)
    plt.show()
    """


    displacement_time = np.zeros(len(t))
    for i, rho_flattened in enumerate(rho_t):
        rho = rho_flattened.reshape(H.Ns, H.Ns)
        mat_3 = displacement.toarray(time=t[i])
        res = np.matmul(rho,mat_3)

        displacement_time[i]= np.trace(res)


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


    omega_new = omega/omega_0

    displacement_total = factor*displacement_time*Mask
    X_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total)
    P_omega = abs(omega**2*X_omega)**2

    # Plot P_omega
    plt.plot(omega_new,P_omega,linewidth=1.0,label=f"T2:{T2_fs}")
plt.legend()
plt.yscale('log')
plt.ylabel(r'P($\omega$) using displacement')
plt.title(f'P($\Omega$) using LiNE eom δ = {delta}, Δt = {time_step}')
plt.xlim(0,100)
plt.savefig(f'test/P(Omega) using LiNE eom δ = {delta}, Δt = {time_step} comparison new.png')