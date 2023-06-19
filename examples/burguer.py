import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def matrices(nx, dt, dx, nu):
    b = nu/(dx*dx)
    A = lil_matrix((nx, nx),)
    B = lil_matrix((nx, nx))
    for i in range(1, nx - 1):
        A[i, i-1] = 1/6 - b*dt/2
        A[i, i] = 2/3 + b*dt 
        A[i, i+1] = 1/6 - b*dt/2
        B[i, i-1] = 1/6 + b*dt/2 
        B[i, i] = 2/3 - b*dt 
        B[i, i+1] = 1/6 + b*dt/2

    A[0, 0] = 2/3 + b*dt 
    A[nx-1, nx-1] = 2/3 + b*dt
    B[0, 0] = 2/3 - b*dt 
    B[nx-1, nx-1] = 1/6 + b*dt/2
    return A, B

L = 1.0  # Length of the domain
nx = 50  # Number of grid points
T = 1.0  # Final time
nt = 100  # Number of time steps
nu = 0.01  # Viscosity coefficient
degree = 2  # Degree of the Lagrange element

dx = L / (nx - 1)
dt = 0.1

x = np.linspace(0, L, nx)
u = np.sin(np.pi*x)


u_new = np.zeros(nx)
b = nu/(dx*dx)
for n in range(nt):
    A, B = matrices(nx, dt, dx, nu)
    d = B.dot(u)
    for i in range(1, nx - 1):
        A[i, i-1] -= dt/4*u[i-1]/dx
        A[i, i] += dt/4*(u[i-1]- u[i])/dx
        A[i, i+1] += dt/4*u[i+1]/dx
        B[i, i-1] += dt/4*u[i-1]/dx
        B[i, i] -= dt/4*(u[i-1]- u[i])/dx
        B[i, i+1] -= dt/4*u[i+1]/dx

    u_new = spsolve(A, d)
    u = u_new.copy()
print(u_new)

# Plot the final solution
plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
# plt.title('')
plt.show()