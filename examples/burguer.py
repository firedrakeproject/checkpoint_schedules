import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters
L = 1.0  # Length of the domain
nx = 100  # Number of grid points
T = 1.0  # Final time
nt = 100  # Number of time steps
nu = 0.001  # Viscosity coefficient
degree = 2  # Degree of the Lagrange element

# Discretization
dx = L / (nx - 1)
dt = T / nt

# Initialize grid
x = np.linspace(0, L, nx)
u = np.sin(np.pi*x)

# Initialize solution
u_new = np.zeros(nx)

# Assemble the stiffness matrix
K = lil_matrix((nx, nx))
for i in range(1, nx - 1):
    K[i, i-1] = -nu / dx**2
    K[i, i] = 2 * nu / dx**2
    K[i, i+1] = -nu / dx**2

# Time-stepping
for n in range(nt):
    # Compute the new solution at each grid point
    for i in range(1, nx - 1):
        u_new[i] = u[i] - dt / (2 * dx) * (u[i+1]**2 - u[i-1]**2) \
                   - dt * K[i, i] * u[i]
    
    # Update the solution
    u = u_new.copy()

# Plot the final solution
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
plt.title('Burger Equation: Lagrange Finite Element')
plt.show()