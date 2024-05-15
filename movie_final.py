import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


# ------------------------------------------------------------------------
plt.style.use('plot_style.txt')
save_dir = '/users/chessbunny/documents/NM2_HW5/figures'

# ------------------------------------------------------------------------
# Simulation Parameters
n = 100
a = 1
t_0 = 0
t_f = 0.5
L = 1
mu = 0.01
dx = 0.0099
dt = 0.5 * dx**2 / mu 

# Domain Setup
x = np.linspace(0, L, n + 2)
U = np.zeros(n + 2)
S = np.ones(n + 2)

# ------------------------------------------------------------------------
def forward_euler(U, dx, dt, mu, S):
    """ Forward Euler scheme for solving the advection-diffusion equation. """
    U_next = np.copy(U)
    for j in range(1, len(U) - 1):
        U_next[j] = U[j] + dt * (-a / (2 * dx) * (U[j+1] - U[j-1]) +
                                 mu / (dx**2) * (U[j+1] - 2*U[j] + U[j-1]) + S[j])
    return U_next

# ------------------------------------------------------------------------
def update(frame):
    global U

    U = forward_euler(U, dx, dt, mu, S)

fig, ax = plt.subplots()
line, = ax.plot(x, U)

ax.set_xlim(0, L)
ax.set_ylim(-0.1, 1.6)  
ax.set_xlabel('Position (x)')
ax.set_ylabel('Concentration (U)')

# ------------------------------------------------------------------------
def init():
    line.set_ydata(np.zeros(n + 2))
    return line,

def animate(i):
    global U
    U = forward_euler(U, dx, dt, mu, S)
    line.set_ydata(U)
    return line,

fig.suptitle(f'Advection-Diffusion @ grid size $={n}$, $\Delta x ={dx:.5f}$ & $\Delta t = {dt:.5f}$')
fig.subplots_adjust(wspace=0.5, hspace=0.5)
# fig.xlabel('Grid Length')
# fig.ylabel('Limit')

ani = FuncAnimation(fig, animate, init_func=init, frames=int((t_f - t_0) / dt), interval = 1, blit=True)
# plt.show()

# Save the animation
ani.save(os.path.join(save_dir, 'exercise3_3.mp4'), writer='ffmpeg')
