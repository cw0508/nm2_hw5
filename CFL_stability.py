import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded
import os


save_dir = '/users/chessbunny/documents/NM2_HW5/figures'
# ------------------------------------------------------------------------
# CFL Stability
# ------------------------------------------------------------------------
# Constants
length_L = 1
diff_coef_mu = 0.01
source_S = 1.0
a = 1.0
t_0 = 0
t_f = 0.5
mesh_sizes = [50, 100, 200] 


# ------------------------------------------------------------------------
# Forward Euler 1D for analysis
def forward_euler(U, dx, dt, mu, S):
    """ Forward Euler scheme for solving the advection-diffusion equation. """
    U_next = np.copy(U)
    for j in range(1, len(U) - 1):
        U_next[j] = U[j] + dt * (-a / (2 * dx) * (U[j+1] - U[j-1]) +
                                 mu / (dx**2) * (U[j+1] - 2*U[j] + U[j-1]) + S)
    return U_next


# ------------------------------------------------------------------------
# Plotting Function for finding a stable CFL parts
def plot_for_different_dx(length_L, grid_sizes, mu, S, t_f):
    """ Plotting the numerical solution for different grid sizes with subplots. """

    dx_list = []
    dt_list = []

    fig, axes = plt.subplots(nrows=1, ncols=len(grid_sizes), figsize=(15, 5), sharey=True)
    fig.suptitle('Numerical Solution at t = 0.5 for Different Grid Sizes',fontsize=26)

    for idx, n in enumerate(grid_sizes):
        dx = length_L / (n + 1)

        dx_list.append(dx)
        dt = 0.01 

        dt = 0.25 * dx**2 / mu
        dt_list.append(dt)

        # # Adjust dt based on the CFL condition
        # if mu * (dt / dx**2) > 0.5:
        #     dt = 0.5 * dx**2 / mu       #1D
        #     dt = 0.25 * dx**2 / mu      #2D
        #     print(f"Adjusted dt to {dt:.6f} @ grid size = {n} & dx = {dx:.6f} to satisfy CFL condition")
        
        U = np.zeros(n + 2)
        t = 0

        while t <= t_f:
            U = forward_euler(U, dx, dt, mu, S)
            t += dt
        x = np.linspace(0, length_L, n + 2)
        ax = axes[idx]
        ax.plot(x, U, '-o')
        ax.set_title(f'Grid Size = {n}, $\Delta x = {dx:.3f}$, $\Delta t = {dt:.4f}$')
        ax.set_xlabel('x')
        if idx == 0:
            ax.set_ylabel('U(x,t)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig(os.path.join(save_dir, 'Q2_2DCFL_1.png'))
    plt.show()

    return dx_list, dt_list

# Run the experiments
dx_list, dt_list = plot_for_different_dx(length_L, mesh_sizes, diff_coef_mu, source_S, t_f) 

print(dx_list)
print(dt_list)

# ------------------------------------------------------------------------
# Plotting function for |m(θ)| 
# Constants
a = 1.0                         # Advection coefficient
# mu = [0.1, 0.01, 0.0049]        # Diffusion coefficients
# dx = [0.02, 0.0099, 0.00498]    # Spatial resolutions
# dt = [0.01, 0.00490, 0.00124]   # Time steps

mu = [0.1, 0.01, 0.001]
dx = dx_list
dt = dt_list
# dx = [0.005, 0.002, 0.001]
# dt = [0.0012, 0.0003, 0.0001]
mu_conditions = len(mu)
dx_conditions = len(dx)

def plot_m_theta():
    theta = np.linspace(0, 2 * np.pi, 400)  # Theta from 0 to 2π radians
    theta_degrees = np.degrees(theta)  
    fig, axes = plt.subplots(nrows=mu_conditions, ncols=dx_conditions, figsize=(10, 8))
    fig.suptitle('|m(θ)| over 0° ≤ θ ≤ 360°', fontsize=16)  # Adjusted fontsize here

    for i in range(mu_conditions):
        for j in range(dx_conditions):
            current_mu = mu[i]
            current_dx = dx[j]
            current_dt = dt[j]
            L = 1  
            n = int(L / current_dx) - 1  

            m_theta = -1j * current_dt * (a / current_dx) * np.sin(theta) + current_dt * (current_mu / current_dx**2) * (2 * np.cos(theta) - 2)
            magnitude_m_theta = np.abs(m_theta)

            ax = axes[i][j]
            ax.plot(theta_degrees, magnitude_m_theta, label='|m(θ)|')
            ax.hlines(1, xmin=0, xmax=360, colors='red', linestyles='dashed', label='Stability limit (|m(θ)|=1)') 
            ax.set_title(f'$\\mu = {current_mu}$, $\\Delta x = {current_dx:.4f}$, $\\Delta t = {current_dt:.4f}$, n = {n}', fontsize=11)  
            ax.set_xlabel('θ (degrees)', fontsize=8)
            if j == 0:
                ax.set_ylabel('|m(θ)|', fontsize=8)
            ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(os.path.join(save_dir, 'Q2_2Dm_theta_1.png'))
    # plt.show()


# ------------------------------------------------------------------------
# Parameters
mu = 1.0          # Diffusion coefficient
dx = 0.1          # Spatial step size
theta = np.linspace(0, 2 * np.pi, 400)  # Wave number array
S = 0             # Source term, assuming zero for simplicity

# Compute the amplification factor g
g_numer = 1 +  1j * np.sin(theta) + S
g_denom = 1 - mu * (2 / dx) * (np.cos(theta) - 1)
g = g_numer / g_denom

# Compute the magnitude of g
g_magnitude = np.abs(g)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(theta, g_magnitude, label='|m(θ)|', color='blue')
plt.axhline(1, color='red', linestyle='--', label='Stability Threshold |m(θ)|=1')
plt.title('Amplification Factor Magnitude vs. Theta')
plt.xlabel('Theta (rad)')
plt.ylabel('|m(θ)|')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Q4_stab.png'))
plt.show()

# Check stability
if np.all(g_magnitude <= 1):
    print("The scheme is stable for the given parameters.")
else:

