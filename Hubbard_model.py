from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian, exp_op, quantum_operator
from quspin.tools.measurements import obs_vs_time
import numpy as np
from numpy.random import uniform, choice
import matplotlib.pyplot as plt


# Define model parameters
a = 4                # Lattice constant in Åcd
NN = 10              # Period of the pulse
omega_0 = 32.9       # Frequency in THz
F_0 = 10             # Amplitude in MV/cm
t_0 = 0.52           # Intra chain hopping parameter in eV
Lx = 5               # system size
N = 2*Lx             # number of particles     
N_up = N // 2 + N%2  # number of spin-up particles
N_down = N // 2


# range in time to evlove system
start,stop,num = 0, 2, 1001      # time in fs
t = np.linspace(start, stop, num=num, endpoint=True)      # Time array
A_t = (F_0*10/omega_0)*((np.sin(omega_0*t/(2*NN)))**2)*np.sin(omega_0*t)     # Vector Potential


# Plot Vector potential; A(t)
plt.plot(t,A_t)
plt.xlabel("t(fs)")
plt.ylabel("A(t)")
plt.show()

# Setting up the basis
basis = spinful_fermion_basis_1d(Lx, Nf=(N_up,N_down))

# create the model
hopping_left = [[-t_0, i, i+1]for i in range(Lx-1)] 
hopping_right = [[+t_0, i, i+1]for i in range(Lx-1)] 

operator_list_0 = [
    ["+-|",hopping_left],       # up hop left
    ["-+|",hopping_right],      # up hop right
    ["|+-",hopping_left],       # down hop left
    ["|-+",hopping_right]       # down hop right
]

current_list =[
    ["+-|",hopping_left],       # up hop left
    ["-+|",hopping_right],      # up hop right
    ["|+-",hopping_left],       # down hop left
    ["|-+",hopping_right]       # down hop right
]

# create operator dictionary for quantum_operator class

operator_dict = dict(H0=operator_list_0)


# set up hamiltonian dictionary and observable (current J)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
H_dict = quantum_operator(operator_dict, basis=basis, **no_checks)

#J = hamiltonian(current_list,)


print(np.arange(N)%Lx)
'''

# Setting uo user-defined BASIC symmetry transformations for 2d lattice 
x = np.arange(N_2d) % Lx  # x positions for sites for one spin species
y = np.arange(N_2d) // Lx # y positions for sites for one spin species
t_x = (x+1) % Lx + Lx * y # translation along x-direction for one spin species 
t_y =  x +Lx * ((y+1)%Ly) # translation along y-direction for one spin species



# create the spin-up spin-down combined transformations
s = np.arange(2*N_2d) 
print(s)

T_x = np.hstack((t_x,t_x+N_2d)) # translation along x-direction for both spin species 
T_y = np.hstack((t_y,t_y+N_2d)) # translation along y-direction for both spin species
#print(T_x)
#print(T_y)

S = np.roll(s,N_2d)
print(S)


# Setting up bases 
basis_2d = spinful_fermion_basis_general(N_2d, simple_symm=False, Nf=(2,2), kxblock=(T_x,0), kyblock=(T_y,0),sblock=(S,0))

# setting up hamiltonian
# setting up ste-coupling lists for advanced case 
hopping_left = [[-t_0, i, T_x[i]]for i in range(2*N_2d)] + [[-t_0,i,T_y[i]]for i in range(2*N_2d)]
hopping_right = [[+t_0, i, T_x[i]]for i in range(2*N_2d)] + [[+t_0,i,T_y[i]]for i in range(2*N_2d)]

static =[
    ["+-",hopping_left],    # spin-up and spin-down hop to left
    ["-+",hopping_right],   # spin-up and spin-down hop to right
]
dynamic = []
# build Hamiltonian 
H = hamiltonian(static,dynamic,basis=basis_2d, dtype=np.float64)

E = H.eigvalsh()
print(E)

E,V = H.eigh()
print(E)
print(V)




###### setting up hamiltonian ######
# setting up site-coupling lists for advanced case
hopping_left = [[-J, i, T_x[i]] for i in range(2 * N_2d)] + [
    [-J, i, T_y[i]] for i in range(2 * N_2d)
]
hopping_right = [[+J, i, T_x[i]] for i in range(2 * N_2d)] + [
    [+J, i, T_y[i]] for i in range(2 * N_2d)
]
potential = [[-mu, i] for i in range(2 * N_2d)]
interaction = [[U, i, i + N_2d] for i in range(N_2d)]
#
static = [
    ["+-", hopping_left],  # spin-up and spin-down hop to left
    ["-+", hopping_right],  # spin-up and spin-down hop to right
    ["n", potential],  # onsite potential for spin-up and spin-down
    ["nn", interaction],
]  # spin-up spin-down interaction
# build hamiltonian
H = hamiltonian(static, [], basis=basis_2d, dtype=np.float64)
# diagonalise H
E = H.eigvalsh()
'''