from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt


# Define model parameters
L = 10           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0.15      # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
NN = 5                               # Period of the pulse
omega_0 = 0.0075                     # Frequency in THz
F_0 = 0.2                            # Amplitude 
tf = 2*np.pi*NN/omega_0              # Final time
t_conversion = 2.4188843265864e-2    # Conversion of time from a.u to fs


# Define atomic-site positions
positions = np.zeros(L)
for i in range(L):
    positions[i]=((i+1)-((L+1)/2))*a-((-1)**(i+1))*delta


# Define the hopping elements using the distances between atoms
#v = -np.exp(-(a-2*delta)) # intracell hopping parameter
#w = -np.exp(-(a+2*delta)) # intercell hopping parameter
v = 1.1
w = 0.9

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

print(hop_pm)
print(hop_mp)

# Define the time array and the Vector potential 
start,stop,num = 0, tf, 250  # time in fs
t = np.linspace(start, stop, num=num, endpoint=False)         # Time array
A_t = F_0*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)   # Vector Potential


# Plot Vector potential A(t)
plt.plot(t*t_conversion,A_t)
plt.title("Vector Potential vs Time(fs)")
plt.xlabel("time (fs)")
plt.ylabel("Vector Potential A(t)")
#plt.show()

# Define time dependent part in the Hamiltonian
def ramp(t,F_0,omega_0,NN):
    A_t = F_0*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t) 
    return np.exp(-1j*A_t)

def ramp_conj(t,F_0,omega_0,NN):
    A_t = F_0*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t) 
    return np.exp(1j*A_t)

ramp_args = [F_0,omega_0,NN,a]


print(ramp_args)
plt.plot(t,ramp(t,F_0,omega_0,NN))
#plt.show()


## Construct single-particle Hamiltonian 

# define basis
basis = spinless_fermion_basis_1d(L, Nf=1)

# define static and dynamic lists and build real-space Hamiltonian
static = [["+-", hop_pm], ["-+", hop_mp]]
dynamic = []
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
E,V = H.eigsh(time=0.0, k=1, which='SA')


# define the Hamitonian in the presence of the external field
stat = []
dyna = [["+-", hop_pm,ramp,ramp_args], ["-+", hop_mp,ramp_conj,ramp_args]]
H_t = hamiltonian(stat,dyna,basis=basis, dtype=np.float64)

# Define the initial state
E_0 , psi_0 = H_t.eigsh(time=0.0, k=1,which="SA")
E,V =H_t.eigh(time=0)
psi_0 = psi_0[:,0]


# Evolve state in 
psi_t = H_t.evolve(psi_0,0,t,eom='SE',iterate=True)
# print(psi_t)

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
#print(np.size(current_time))

plt.figure()
plt.plot(t,current_time)
plt.xlabel('Time (fs)')
plt.ylabel('Current')
plt.title("Current vs Time")
plt.show()


# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]

for k in range(num):
    current_time[k] = np.exp(1j*omega[0]*k*delta_t)*current_time[k]

J_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.fft(current_time)
omega_J_omega = omega * J_omega
S_omega = abs(omega_J_omega)**2

for k in range(num):
    A_t[k] = np.exp(1j*omega[0]*k*delta_t)*A_t[k]

A_t_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.fft(A_t)
plt.plot(omega,np.abs(A_t_omega)**2)
plt.show()


# Plot J(t) and omega * J(omega) (magnitude)
plt.figure()
plt.plot(omega,S_omega)
plt.yscale('log')
plt.ylabel("S(Omega)")
plt.show()

W = sp.signal.windows.cosine(num)
J_omega_W = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.fft(current_time*W)
omega_J_omega_W = omega * J_omega_W
S_omega_W = abs(omega_J_omega_W)**2


# Plot J(t) and omega * J(omega) (magnitude)
plt.figure()
plt.plot(S_omega_W)
plt.yscale('log')
plt.ylabel("S(Omega)_W")
plt.show()

