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
import csv
import os

###########################################
######  Crank-Nicolson Propagation  #######
###########################################

# Create output directory if not exists
output_dir = "Crank-Nicolson Propagation"
os.makedirs(output_dir, exist_ok=True)

# Define model parameters
L = 100           # system size
J = 1.0           # uniform hopping contribution
a = 2.0           # Lattice constant in a.u
delta_values = [0.15 , - 0.15]     # Alternating shift of the atos casuing the dimerization (negative for topological phase)


# Declare constants for Vector Potential
N_cyc = 5                            # Period of the pulse
omega_0 = 0.0075                      # Frequency in a.u
A_0 = 0.2                            # Amplitude 
tf = 2*np.pi*N_cyc/omega_0           # Final time
t_conversion = 2.4188843265864e-2    # Conversion of time from a.u to fs


# Open CSV file
csv_filename = os.path.join(output_dir, "P_omega_combined.csv")
with open(csv_filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["omega"] + [f"P_omega_delta_{delta}" for delta in delta_values])

    P_omega_data = []
    for delta in delta_values:
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
                hop_pm_v = hop_pm_v + [[-v, i, (i+1)%L]] #i+1
                hop_mp_v = hop_mp_v + [[v, i, (i+1)%L]]
            else:
                hop_pm_w = hop_pm_w + [[-w, i, (i+1)%L]]
                hop_mp_w = hop_mp_w + [[w, i, (i+1)%L]]


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


        ## Construct single-praticle Hamiltonian
        # define basis
        basis = spinless_fermion_basis_1d(L, Nf=1)

        # define the Hamitonian in the presence of the external field
        stat = []
        dyna = [["+-", hop_pm_v,ramp_v,ramp_args], 
                ["+-", hop_pm_w,ramp_w,ramp_args],
                ["-+", hop_mp_v,ramp_v_conj,ramp_args],
                ["-+", hop_mp_w,ramp_w_conj,ramp_args]]

        H_t = hamiltonian(stat,dyna,basis=basis, dtype=np.float64)

        # Solving the time-dependent Hamiltonian to obatin L/2 number of eigenstates correspond to lowest eigenvalues
        E , V = H_t.eigsh(time=0.0, k=L/2,which="SA")

        delta_t = t[1]-t[0]   # Time step
        I = np.eye(L)

        # Define time-evolution funtion
        def evolve_state(t,state):
            H = H_t.toarray(time=t+delta_t/2)
            mat_a = I+1j*H*delta_t/2
            return np.linalg.solve(mat_a, state-np.dot(1j*H*delta_t/2,state))

        displacement_time = np.zeros(len(t))

        # Calculate expectation value of displacement
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
        plt.title(f"Displacement vs Time δ = {delta}")
        plt.savefig(f'Crank-Nicolson Propagation/Displacement δ = {delta}, Δt = {time_step}.png')

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

        # Fourier Transforamtion of dispalcement and acceleration
        displacement_time = factor*displacement_time*Mask
        X_omega = (delta_t/(np.sqrt(2*np.pi)))*N*sp.fft.ifft(displacement_time)
        P_omega = abs(omega**2*X_omega)**2

        P_omega_data.append(P_omega)

        ## Plot High Harmonic Spectra
        plt.figure(figsize=(8, 6))
        plt.plot(omega_new,P_omega,linewidth=1.0)
        plt.yscale('log')
        plt.ylabel(r'P($\omega$)')
        plt.xlabel(r'$\omega/\omega_0$')
        plt.title(f'P($\omega$) using Crank-Nicolson method including PBC δ = {delta}')
        plt.xlim(0,100)
        plt.savefig(f'Crank-Nicolson Propagation/P(omega) using Crank-Nicolson method including PBC δ = {delta}, Δt = {time_step}.png')
    
    # Write data to CSV file
    for i in range(len(omega)):
        writer.writerow([omega_new[i]] + [P_omega_data[j][i] for j in range(len(delta_values))])


## Plot High Harmonic Spectra
colors = ['r','k','g']
plt.figure(figsize=(10, 8))
for i, delta in enumerate(delta_values):
    plt.plot(omega / omega_0, P_omega_data[i], label=f"$\delta$={delta}",linewidth=1.0, color=colors[i])

plt.ylabel(r'P($\omega$)')
plt.xlabel(r'$\omega/\omega_0$')
plt.title(f"High Harmonic Spectrum for Different Delta Values including PBC Δt ={time_step}")
plt.yscale('log')
plt.legend()
plt.xlim(0,100)
plt.savefig(os.path.join(output_dir, f"P_omega_comparison including PBC Δt ={time_step}.png"))