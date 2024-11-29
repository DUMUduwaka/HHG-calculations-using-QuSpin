from quspin.operators import hamiltonian            # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space fermion basis
from quspin.tools.measurements import obs_vs_time   # Tools for measurements
import scipy as sp
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta = 0.15     # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
N_cyc = 5                            # Period of the pulse
omega_0 = 0.0075                      # Frequency in a.u
A_0 = 0.2                            # Amplitude 
tf = 2*np.pi*N_cyc/omega_0           # Final time
t_conversion = 2.4188843265864e-2    # Conversion of time from a.u to fs

print("final time",tf)
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
start,stop,num = 0, tf, 4200                                  # time in fs
t = np.linspace(start, stop, num=num, endpoint=False)         # Time array
A_t = A_0*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)   # Vector Potential


# Plot Vector potential A(t)
plt.figure()
plt.plot(t*t_conversion,A_t)
plt.title("Vector Potential vs Time(fs)")
plt.xlabel("time (fs)")
plt.ylabel("Vector Potential A(t)")
plt.savefig('Plots/Vector Potential.png')


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


# Define correct current operator

'''

delta_x_v = []
delta_x_w = []

for i in range(L-1):
    if i%2 == 0:
        delta_x_v = delta_x_v + [[positions[1]-positions[0],i,i+1]]
    
    else:
        delta_x_w = delta_x_w + [[positions[2]-positions[1],i,i+1]]


def current_ramp_v(t,A_0,omega_0,N_cyc,a):
    A_t = (A_0/omega_0)*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)
    return 0.5*(1j*delta_x_v - delta_x_v**2*A_t)

def current_ramp_w(t,A_0,omega_0,N_cyc,a):
    A_t = (A_0/omega_0)*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)
    return 0.5*(1j*delta_x_w - delta_x_w**2*A_t)

def current_ramp_v_conj(t,A_0,omega_0,N_cyc,a):
    A_t = (A_0/omega_0)*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)
    return 0.5*(-1j*delta_x_v - delta_x_v**2*A_t)

def current_ramp_w_conj(t,A_0,omega_0,N_cyc,a):
    A_t = (A_0/omega_0)*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)
    return 0.5*(-1j*delta_x_w - delta_x_w**2*A_t)

current_ramp_args = [A_0,omega_0,N_cyc,a]
current_static = []
current_dynamic = [["+-", hop_pm_v,current_ramp_v,current_ramp_args],
                   ["+-", hop_pm_w,current_ramp_w,current_ramp_args], 
                   ["-+", hop_mp_v,current_ramp_v_conj,current_ramp_args],
                   ["-+", hop_mp_w,current_ramp_w_conj,current_ramp_args]]

'''


# Define current operator
def current_ramp(t,A_0,omega_0,N_cyc,a):
    A_t = (A_0/omega_0)*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)
    return -1j*a*np.exp(-1j*A_t)

def current_ramp_conj(t,A_0,omega_0,N_cyc,a):
    A_t = (A_0/omega_0)*((np.sin(omega_0*t/(2*N_cyc)))**2)*np.sin(omega_0*t)
    return 1j*a*np.exp(1j*A_t)

current_ramp_args = [A_0,omega_0,N_cyc,a]

#delat_x_i = [[positions[i],i] for i in range(L)]

current_static = []
current_dynamic = [["+-", hop_pm_v,current_ramp,current_ramp_args],
                   ["+-", hop_pm_w,current_ramp,current_ramp_args], 
                   ["-+", hop_mp_v,current_ramp_conj,current_ramp_args],
                   ["-+", hop_mp_w,current_ramp_conj,current_ramp_args]]



current = hamiltonian(current_static,current_dynamic, basis=basis,dtype=np.float64)


# Define position operator

X_i = [[positions[i],i] for i in range(L)]

displacement_static=[["n",X_i]]
displacement_dynamic=[]
displacement = hamiltonian(displacement_static,displacement_dynamic,basis=basis, dtype=np.float64)


E , V = H_t.eigsh(time=0.0, k=L/2,which="SA")
print("Eigen States are",V)
print("Eigen Energies are",E)

# Store eigenstates in a list
eigenstates = [V[:,i] for i in range(int(L/2))]

current_total = np.zeros(len(t))
displacement_total = np.zeros(len(t))

for i, psi_0 in enumerate(eigenstates):
    print(i)
    
    #print(f"Eigenstate {i}:")
    #print(psi_0)

    # Evolve state 
    psi_t = H_t.evolve(psi_0,0,t,eom='SE',iterate=True)
   
    Obs_time = obs_vs_time(psi_t,t,dict(Energy=H_t,Current=current, Displacement=displacement))
    current_time = Obs_time["Current"]
    displacement_time = Obs_time["Displacement"]

    #print(f"Displacement {i}:")
    #print(displacement_time)
    
    current_total =+ current_time
    displacement_total =+ displacement_time 
    
velocity_total = np.gradient(displacement_total,t)

plt.figure()
plt.plot(t*t_conversion,current_total)
plt.xlabel('Time (fs)')
plt.ylabel('Current')
plt.title("Current vs Time")
plt.savefig('Plots/Current.png')


plt.figure()
plt.plot(t*t_conversion,displacement_total)
plt.xlabel('Time (fs)')
plt.ylabel('Displacement')
plt.title("Displacement vs Time")
plt.savefig('Plots/Displacement.png')

plt.figure()
plt.plot(t*t_conversion,velocity_total)
plt.xlabel('Time (fs)')
plt.ylabel('Current')
plt.title("velocity vs Time")
plt.savefig('Plots/velocity.png')


# Fourier Transformation to obtain current in frequency domain
T = t[-1]-t[0]        # total time 
N = num               # Number of steps
delta_t = t[1]-t[0]   # Time step


omega = np.linspace(-1*np.pi/delta_t, np.pi/delta_t, num, endpoint= False)
delta_omega=omega[1]-omega[0]

factor = np.zeros(num)
for k in range(num):
    factor[k] = np.exp(1j*omega[0]*k*delta_t)

current_total = factor*current_total
J_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(current_total)
omega_J_omega = omega * J_omega
S_omega = abs(omega_J_omega)**2


displacement_total = factor*displacement_total
X_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_total)
P_omega = abs(omega**2*X_omega)**2

current_total_total = factor*current_total
V_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(current_total)
Pv_omega = abs(omega*X_omega)**2

A_t_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(factor*A_t)
plt.figure()
plt.plot(omega,np.abs(A_t_omega)**2)
plt.yscale('log') 
plt.title("FFT of AT")
plt.xlim(left=0)
plt.savefig('Plots/FFT of A(t).png')


plt.figure()
plt.plot(omega,np.abs(X_omega)**2)
plt.yscale('log')
plt.title("FFT of Xt")
plt.xlim(left=0)
plt.savefig('Plots/FFT of X(t).png')


omega_new = omega/omega_0
# Plot J(t) and omega * J(omega) (magnitude)
plt.figure()
plt.plot(omega_new,S_omega)
plt.yscale('log')
plt.ylabel(r'S($\omega$)')
plt.xlim(0,60)
plt.savefig('Plots/S(Omega).png')

# Plot P_omega
plt.figure()
plt.plot(omega_new,P_omega)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.xlim(0,60)
plt.savefig('Plots/P(Omega).png')

# Plot Pv_omega
plt.figure()
plt.plot(omega_new,Pv_omega)
plt.yscale('log')
plt.ylabel(r'P($\omega$)')
plt.xlim(0,60)
plt.savefig('Plots/Pv(Omega).png')


'''
# Use density matrix to get harmonic spectrum
Rho = np.zeros((L,L))

print((V[:,1]))

print(np.transpose(np.conjugate(V[:,1])))




eigenstates = [V[:,i] for i in range(int(L/2))]
print(np.shape(eigenstates))

for i in range(int(L/2)):
    Rho =+ np.dot(V[:,i],np.transpose(np.conjugate(V[:,i])))

print((Rho))





#Evolve Density matrix
Rho_t = H_t.evolve(Rho, 0,t, eom='LvNE',iterate=True)
print(type(Rho_t))
Rho_list = list(Rho_t)
print(Rho_list[0])

current_dense=current.toarray(time=t[1])
print("current is :",current_dense)

current_time_den_mat=np.zeros(len(t))
for i in range(len(t)):
    mat_1 = Rho_list[i]
    mat_2 = current.toarray(time=t[i])
    res = np.dot(mat_1,mat_2)
    current_time_den_mat[i]= np.trace(res)

#print("Current using the density matrix ", current_time_den_mat)
plt.figure()
plt.plot(t*t_conversion,current_time_den_mat)
plt.xlabel('Time (fs)')
plt.ylabel('Current')
plt.title("Current vs Time using density matrix")
plt.savefig('Plots/Current_density_mat.png')

current_time_den_mat = factor*current_time_den_mat
J_omega_den_mat = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(current_time_den_mat)
omega_J_omega_den_mat = omega * J_omega_den_mat
S_omega_den = abs(omega_J_omega_den_mat)**2

plt.figure()
plt.plot(omega,S_omega_den)
plt.yscale('log')
plt.ylabel(r'S($\omega$)')
plt.title(r'S($\omega$) using the density matrix')
plt.xlim(left=0)
plt.savefig('Plots/S(Omega)_density_matrix.png')
'''
