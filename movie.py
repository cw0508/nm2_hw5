import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_banded
import os 

save_dir = '/users/chessbunny/documents/NM2_HW5/figures'

a_x, a_y, a_z = 1.0, 1.0, 1.0  # Advection coefficients

# Simulation parameters
N = 200
dx, dy = 0.0050, 0.0050
dt = 0.0006
mu = 1
mus = [1, 0.1, 0.01, 0.001]  # Different values of mu for each subplot
frames = 200

# ------------------------------------------------------------------------
# Forward Euler 2D for animations
def forward_euler_2d(U, dx, dy, dt, mu, S):
    U_next = np.copy(U)
    n, m = U.shape
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            advection_x = (U[i+1, j] - U[i-1, j]) / (2 * dx)
            advection_y = (U[i, j+1] - U[i, j-1]) / (2 * dy)
            diffusion_x = (U[i+1, j] - 2*U[i, j] + U[i-1, j]) / (dx**2)
            diffusion_y = (U[i, j+1] - 2*U[i, j] + U[i, j-1]) / (dy**2)
            U_next[i, j] = U[i, j] + dt * (-a_x * advection_x - a_y * advection_y + mu * (diffusion_x + diffusion_y) + S[i, j])
    
    return U_next


# ------------------------------------------------------------------------
def tridiagonal_solver(mu, dt, dx, F, nx):
    c = mu * (dt / dx**2 + dt/dx**2)
    ab = np.zeros((3, nx))          # nx, not nx+2, assuming F is of length nx and boundary conditions are handled outside
    ab[0, 1:] = -c                  # Upper diagonal
    ab[1, :] = 1 + 2*c              # Main diagonal
    ab[2, :-1] = -c                 # Lower diagonal
    V = solve_banded((1, 1), ab, F)  # Ensure F is of length nx
    return V

def forward_euler_2d_tri(U, dx, dy, dt, mu, S):
    nx, ny = U.shape[0] - 2, U.shape[1] - 2
    U_next = np.copy(U)
    for i in range(1, nx+1):
        F = U[i, 1:-1] + dt * S[i, 1:-1] 
        U_next[i, 1:-1] = tridiagonal_solver(mu, dt, dx, F, ny)
    return U_next


# ------------------------------------------------------------------------
# Single movie:
# Initial condition and source setup
U = np.zeros((N+2, N+2))
S = np.ones((N+2, N+2))

fig, ax = plt.subplots()
cax = ax.matshow(U, cmap='coolwarm')
cbar = fig.colorbar(cax)
cbar.set_label('Concentration level')

def update(frame):
    global U

    U = forward_euler_2d_tri(U, dx, dy, dt, mu, S)

    cax.set_data(U)
    ax.set_title(f'Advection-Diffusion @ grid size $={N}$, $\mu = {mu}$, $\Delta x ={dx}$ & $\Delta t = {dt}$')

    return cax,

ani = FuncAnimation(fig, update, frames=frames, blit=True)
# ani.save(os.path.join(save_dir, 'adv_diff_ss3.mp4'), writer='ffmpeg')
# plt.show()


# ------------------------------------------------------------------------
# Three movie:
Us = [np.zeros((N+2, N+2)) for _ in mus]
S = np.ones((N+2, N+2))

# Create figure and axes for the subplots
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Generate initial plots for each value of mu
caxs = [ax.matshow(Us[i], cmap='coolwarm') for i, ax in enumerate(axes)]

# Add colorbars for each subplot, adjusting their size and padding
cbars = [fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04) for cax, ax in zip(caxs, axes)]

for cbar in cbars:
    cbar.set_label('Concentration level')

def update(frame):
    for i, (ax, cax, mu) in enumerate(zip(axes, caxs, mus)):
        Us[i] = forward_euler_2d_tri(Us[i], dx, dy, dt, mu, S)
        cax.set_data(Us[i])
        ax.set_title(f'μ = {mus[i]}')
    return caxs

# fig.suptitle(f'Advection-Diffusion Simulation @ grid size $={N}$, $\Delta x ={dx}$ & $\Delta t = {dt}$')
fig.suptitle(f'Advection-Diffusion Simulation @ grid size $={N}$, $\Delta x ={dx}$ & $\Delta t = {dt}$', y=0.85)

fig.subplots_adjust(wspace=0.5, hspace=0.5)

ani = FuncAnimation(fig, update, frames=frames, blit=True)
ani.save(os.path.join(save_dir, 'adv_diff_Q52.mp4'), writer='ffmpeg')
plt.show()




