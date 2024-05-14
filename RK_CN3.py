import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

save_dir = '/users/chessbunny/documents/NM2_HW5/figures'
plt.style.use('plot_style.txt')

# -----------------------------------------------------------------
def exact_solution(x, t, mu):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * mu * t)

def source_term(x, t, a, mu):
    return a * np.pi * np.cos(np.pi * x) * np.exp(-np.pi**2 * mu * t)

def run_simulation(n, L, mu, a, t_final):
    dx = L / (n + 1)
    dt = 0.25 * dx**2 / mu
    x = np.linspace(dx, L - dx, n)
    u_numerical = np.zeros(n)
    
    def advection_operator(u, dx, a):
        dudx = np.zeros_like(u)
        dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        return -a * dudx

    def diffusion_operator(u, dx, mu):
        d2udx2 = np.zeros_like(u)
        d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        return mu * d2udx2

    def runge_kutta_step(u, dx, dt, a):
        k1 = advection_operator(u, dx, a)
        k2 = advection_operator(u + 0.5 * dt * k1, dx, a)
        k3 = advection_operator(u + 0.5 * dt * k2, dx, a)
        k4 = advection_operator(u + dt * k3, dx, a)
        return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def crank_nicolson_step(u, dx, dt, mu, n):
        diagonal = np.ones(n) * (1 + mu * dt / dx**2)
        off_diagonal = np.ones(n - 1) * (-mu * dt / (2 * dx**2))
        A = diags([diagonal, off_diagonal, off_diagonal], [0, -1, 1]).tocsc()
        B_u = u + 0.5 * dt * diffusion_operator(u, dx, mu)
        return spsolve(A, B_u)

    for step in range(int(t_final / dt)):
        current_time = step * dt
        S = source_term(x, current_time, a, mu)
        u_numerical = runge_kutta_step(u_numerical, dx, dt, a)
        u_numerical = crank_nicolson_step(u_numerical + dt * S, dx, dt, mu, n)

    u_exact = exact_solution(x, t_final, mu)
    error = np.linalg.norm(u_numerical - u_exact, 2) / np.linalg.norm(u_exact, 2)
    return dx, error

# -----------------------------------------------------------------
# Order of accuracy: 
# Parameters
L = 1.0
a = 1.0
t_final = 0.5

# Different grid resolutions
resolutions = [50, 100, 200]
errors = []
dx_values = []

# Explore different mu values and possibly adjust dt accordingly
mu_values = [0.1, 0.01, 0.001] 

fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 

for index, mu in enumerate(mu_values):
    errors = []
    dx_values = []
    for n in resolutions:
        dx, error = run_simulation(n, L, mu, a, t_final)
        errors.append(error)
        dx_values.append(dx)

    # Perform linear regression to calculate the slope
    coefficients = linregress(np.log(dx_values), np.log(errors))
    slope = coefficients.slope

    legend_properties = {'weight':'bold'}

    # Plot each mu value's error
    axs[index].loglog(dx_values, errors, 'o-', base=10, label=f'μ={mu}')
    axs[index].plot(dx_values, np.exp(coefficients[1]) * np.array(dx_values)**slope, 'r--', label=f'Fit Line (Slope: {slope:.2f})')
    axs[index].set_title(f'Convergence Study for μ={mu}')
    axs[index].set_xlabel('Grid Spacing Δx')
    axs[index].set_ylabel('Relative L2 Error')
    axs[index].grid(True, which="both", ls="--")
    axs[index].legend(prop=legend_properties)

plt.tight_layout()
# plt.savefig(os.path.join(save_dir, 'adv_diff_Q6_1.png'))
# plt.show()


# -----------------------------------------------------------------
# m(\theta)
# Parameters
a = 1.0
L = 1.0
t_final = 0.5
resolutions = [50, 100, 200]
mu_values = [0.1, 0.01, 0.001]

def m_theta(theta, a, mu, dx, dt):
    m_adv = 1 - 1j * (a * dt / dx) * np.sin(theta) 
    m_diff = (1 - (mu * dt / dx**2) * (1 - np.cos(theta))) / (1 + (mu * dt / dx**2) * (1 - np.cos(theta)))
    return m_adv * m_diff

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
theta = np.linspace(0, 2 * np.pi, 1000)

for i, mu in enumerate(mu_values):
    dx = L / (201)  
    dt = 0.25 * dx**2 / mu 

    m_values = m_theta(theta, a, mu, dx, dt)
    m_magnitude = np.abs(m_values)
    legend_properties = {'weight':'bold'}

    axs[i].plot(theta, m_magnitude, label=f'|m(θ)|, μ={mu}')
    axs[i].axhline(1, color='red', linestyle='--', label='Stability threshold (|m(θ)|=1)')
    axs[i].set_title(f'Von Neumann Stability for μ={mu}')
    axs[i].set_xlabel('θ (radians)')
    axs[i].legend(prop=legend_properties)
    axs[i].grid(True)

axs[0].set_ylabel('|m(θ)|')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adv_diff_Q6_2.png'))
plt.show()
