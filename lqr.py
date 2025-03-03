import numpy as np
import scipy.linalg

# Define the system dynamics
theta0 = 0.1  # Linearizing around theta = 0
v0 = 0.1      # Nominal velocity

A = np.array([[0, 0, -v0 * np.sin(theta0)],
              [0, 0,  v0 * np.cos(theta0)],
              [0, 0,  0]])

B = np.array([[np.cos(theta0), 0],
              [np.sin(theta0), 0],
              [0, 1]])

# Define cost matrices
Q = np.diag([1, 1, 10])  # Penalizing position errors and large heading errors
R = np.diag([5, 5])      # Penalizing control effort

# Solve Riccati equation for P
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
print(P)

# Compute LQR gain
K = np.linalg.inv(R) @ B.T @ P

print("LQR Gain Matrix K:")
print(K)

# Apply LQR control
def lqr_control(state, reference):
    error = state - reference  # Compute error
    u = -K @ error  # Compute control action
    return u

# Example state
state = np.array([-0.5, -0.5, 0.1])  # (x, y, theta) deviation from reference
reference = np.array([0, 0, 0])    # Desired reference state

# Compute control input
u = lqr_control(state, reference)
print("Control Input (v, omega):", u)
