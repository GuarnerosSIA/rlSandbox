import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Pendulum parameters
g = 9.81  # gravity (m/s^2)
L = 0.5   # length of the pendulum (m)

# Define the system of ODEs
def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta) - 0.1*omega
    return [dtheta_dt, domega_dt]

# Initial conditions for trajectories
theta0 = np.linspace(-2 * np.pi, 2 * np.pi, 10)  # Various initial angles
omega0 = np.linspace(-5, 5, 10)                  # Various initial angular velocities

# Create a grid of initial conditions
theta0, omega0 = np.meshgrid(theta0, omega0)

# Plot trajectories
plt.figure(figsize=(10, 6))
for i in range(theta0.shape[0]):
    for j in range(theta0.shape[1]):
        y0 = [theta0[i, j], omega0[i, j]]
        sol = solve_ivp(pendulum, [0, 50], y0, t_eval=np.linspace(0, 50, 600))
        plt.plot(sol.y[0], sol.y[1], 'b-', linewidth=0.5)

# Add equilibrium points
plt.plot(0, 0, 'ro', label='Stable Equilibrium (0, 0)')
plt.plot(np.pi, 0, 'go', label='Unstable Equilibrium ($\pi$, 0)')

# Customize plot
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$\omega$ (rad/s)')
plt.title('Phase Diagram of a Pendulum')
plt.grid()
plt.legend()
plt.xlim(-2 * np.pi, 2 * np.pi)
plt.ylim(-5, 5)
plt.show()