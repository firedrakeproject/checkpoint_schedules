import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class burger_equation():
    def __init__(self, L, nx, dt, T, nu):
        self.nt = int(T/dt)
        self.nu = nu
        self.dx = L / (nx - 1)
        self.dt = dt
        self.nx = nx

    def forward(self, u0, t0, tf):
        """Execute the foward system in time.

        Paramters
        ---------
        u0 : 
            Forward initial condition.
        t0 : int
            Initial step.
        tn : int
            Final step.
        """
        dx = self.dx
        nx =  self.nx
        dt =  self.dt
        b = self.nu/(dx * dx)
        u = u0
        u_new = np.zeros(nx)
        b = self.nu/(dx*dx)
        steps = int(tf - t0)
        t = 0
        while t < steps:
            # Assemble of the mattrices system
            A = lil_matrix((nx, nx),)
            B = lil_matrix((nx, nx))
            A[0, 0] = 2/3 + b * dt 
            A[self.nx - 1, nx - 1] = 2/3 + b * dt
            B[0, 0] = 2/3 - b*dt
            B[nx - 1, nx - 1] = 1/6 + b*dt/2
            for i in range(1, nx - 1):
                A[i, i-1] = 1/6 - b * dt/2 - dt/4*u[i-1]/dx
                A[i, i] = 2/3 + b * dt + dt/4*(u[i-1]- u[i])/dx
                A[i, i+1] = 1/6 - b*dt/2 + dt/4*u[i+1]/dx
                B[i, i-1] = 1/6 + b*dt/2 + dt/4*u[i-1]/dx
                B[i, i] = 2/3 - b*dt - dt/4*(u[i-1]- u[i])/dx
                B[i, i+1] = 1/6 + b*dt/2 - dt/4*u[i+1]/dx
            d = B.dot(u)
            u_new = spsolve(A, d)
            u = u_new.copy()
            t += 1

        # Plot the final solution
        plt.plot(x, u)
        plt.xlabel('x')
        plt.ylabel('u')
        # plt.title('')
        plt.show()


    def backward(self, u_fwd, p):
        """Execute the adjoint system in time.

        Paramters
        ---------
        u0 : 
            Forward initial condition.
        t0 : int
            Initial step.
        tn : int
            Final step.
        """
        dx = self.dx
        nx =  self.nx
        dt =  self.dt
        b = self.nu/(dx * dx)
        u = u0
        u_new = np.zeros(nx)
        b = self.nu/(dx*dx)
        steps = int(tf - t0)
        t = 0
        while t < steps:
            # Assemble of the mattrices system
            A = lil_matrix((nx, nx),)
            B = lil_matrix((nx, nx))
            A[0, 0] = 2/3 + b * dt 
            A[self.nx - 1, nx - 1] = 2/3 + b * dt
            B[0, 0] = 2/3 - b*dt
            B[nx - 1, nx - 1] = 1/6 + b*dt/2
            for i in range(1, nx - 1):
                A[i, i-1] = 1/6 - b * dt/2 - dt/4*u[i-1]/dx
                A[i, i] = 2/3 + b * dt + dt/4*(u[i-1]- u[i])/dx
                A[i, i+1] = 1/6 - b*dt/2 + dt/4*u[i+1]/dx
                B[i, i-1] = 1/6 + b*dt/2 + dt/4*u[i-1]/dx
                B[i, i] = 2/3 - b*dt - dt/4*(u[i-1]- u[i])/dx
                B[i, i+1] = 1/6 + b*dt/2 - dt/4*u[i+1]/dx
            
            d = B.dot(u)
            u_new = spsolve(A, d)
            u = u_new.copy()
            t += 1

        # Plot the final solution
        plt.plot(x, u)
        plt.xlabel('x')
        plt.ylabel('u')
        # plt.title('')
        plt.show()

L = 1
nx = 100
dt = 0.001
T = 1.0
nu = 0.01

burger = burger_equation(L, nx, dt, T, nu)
x = np.linspace(0, L, nx)
u0 =  np.sin(np.pi*x)
f_step = int(T/dt)
i_step = 0
burger.forward(u0, i_step, f_step)