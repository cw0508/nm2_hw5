import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------------------
# Constants (never change)
length_L = 1
length_Lx, length_Ly, length_Lz = 1, 1, 1
a_x, a_y, a_z = 1.0, 1.0, 1.0  # Advection coefficients
source_S = 1.0
a = 1.0
t_0 = 0
t_f = 0.5

# Constants (can change)
nx, ny, nz = 200, 200, 200  # Grid points
dx, dy, dz = 0.0050, 0.0050, 0.0050
dt = 0.0006
mu = 0.01

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
# Movies:
# ------------------------------------------------------------------------
# 2D: 
# Initial condition and source setup
U = np.zeros((nx+2, ny+2))
S = np.ones((nx+2, ny+2)) * 1

fig, ax = plt.subplots()
cax = ax.matshow(U, cmap='coolwarm')
fig.colorbar(cax)

def update(frame):
    global U
    U = forward_euler_2d(U, dx, dy, dt, mu, S)
    cax.set_data(U)
    return cax,

ani = FuncAnimation(fig, update, frames=100, blit=True)
plt.show()




