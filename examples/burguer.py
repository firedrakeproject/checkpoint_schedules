import numpy as np
import functools
import matplotlib.pyplot as plt
from sympy import *
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from checkpoint_schedules import \
     (Forward, EndForward, Reverse, Copy, EndReverse)
from checkpoint_schedules import RevolveCheckpointSchedule, StorageLocation


class burger_equation():
    """This object define the forward burger's equation its
    and adjoint system.

    Attributes
    ----------
    L : float
        The domain lenght
    nx : int
        Number of nodes.
    dt : float
        Time step.
    T : float
        Period of time.
    nu : float
        The viscosity.
    """
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
        u0 : numpy
            Forward initial condition.
        t0 : int
            Initial step.
        tn : int
            Final step.
        """
        dx = self.dx
        nx = self.nx
        dt = self.dt
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
            A[0, 1] = 1/6 - b*dt/2 + dt/4*u[1]/dx
            A[self.nx - 1, nx - 1] = 2/3 + b * dt
            A[self.nx - 1, nx - 2] = 1/6 - b * dt/2 - dt/4 * u[nx - 2]/dx
            B[0, 0] = 2/3 - b*dt
            B[0, 1] = 1/6 + b*dt/2 - dt/4*u[1]/dx
            B[nx - 1, nx - 1] = 2/3 - b * dt
            B[nx - 1, nx - 2] = 1/6 + b * dt/2 + dt/4 * u[nx - 2]/dx
            for i in range(1, nx - 1):
                v_m = u[i]/dx
                v_mm1 = u[i - 1]/dx
                A[i, i - 1] = 1/6 - b * dt/2 - dt/4*v_mm1
                A[i, i] = 2/3 + b * dt + dt/4*(v_mm1 - v_m)
                A[i, i + 1] = 1/6 - b*dt/2 + dt/4*v_m
                B[i, i - 1] = 1/6 + b*dt/2 + dt/4*v_mm1
                B[i, i] = 2/3 - b*dt - dt/4*(v_mm1 - v_m)
                B[i, i + 1] = 1/6 + b*dt/2 - dt/4*v_m
                
            d = B.dot(u)
            u_new = spsolve(A, d)
            u = u_new.copy()
            t += 1

        return u_new

    def backward(self, u_fwd, p0):
        """Execute the adjoint system in time.

        Parameters
        ---------
        u_fwd : numpy
            Forward solution that is the adjoint dependency.
        p0 : numpy
            Adjoint solution used to initialise de adjoint solver.
        """
        dx = self.dx
        nx = self.nx
        dt = self.dt
        b = self.nu/(dx * dx)
        p_new = np.zeros(nx)
        b = self.nu/(dx*dx)

        # Assemble of the mattrices system
        A = lil_matrix((nx, nx),)
        B = lil_matrix((nx, nx))
        A[0, 0] = 2/3 + b * dt 
        A[0, 1] = 1/6 - b*dt/2 + dt/4*u[1]/dx
        A[self.nx - 1, nx - 1] = 2/3 + b * dt
        A[self.nx - 1, nx - 2] = 1/6 - b * dt/2 - dt/4 * u[nx - 2]/dx
        B[0, 0] = 2/3 - b*dt
        B[0, 1] = 1/6 + b*dt/2 - dt/4*u[1]/dx
        B[nx - 1, nx - 1] = 2/3 - b * dt
        B[nx - 1, nx - 2] = 1/6 + b * dt/2 + dt/4 * u[nx - 2]/dx
        for i in range(1, nx - 1):
            v_m = u_fwd[i]/dx
            v_mm1 = u_fwd[i - 1]/dx
            A[i, i - 1] = 1/6 - b * dt/2 - dt/4*v_mm1
            A[i, i] = 2/3 + b * dt + dt/4*(v_mm1 - v_m)
            A[i, i + 1] = 1/6 - b*dt/2 + dt/4*v_m
            B[i, i - 1] = 1/6 + b*dt/2 + dt/4*v_mm1
            B[i, i] = 2/3 - b*dt - dt/4*(v_mm1 - v_m)
            B[i, i + 1] = 1/6 + b*dt/2 - dt/4*v_m
        
        d = B.dot(u)
        p_new = spsolve(A, d)
        return p_new

    def forward_solution(self):
        u = np.zeros((self.nt, self.nx))
        x = np.linspace(0, L, self.nx)
        t0 = np.exp(1/(8*self.nu))
        for step in range(self.nt):
            t = 1 + step*self.dt
            for i in range(self.nx):
                e = np.exp(x[i]*x[i]/(4*self.nu*t))
                a = (t/t0)**(0.5)
                u[step][i] = (x[i]/(t))/(1 + (a*e))

        return u[-1]

    def dudx(self):
        x = Symbol('x')
        t0 = exp(1/(8*self.nu))
        e = exp(x*x/(4*self.nu))
        t = 1
        a = (t/t0)**(0.5)
        u0 = (x/(t))/(1 + (a*e))
        du_dx = diff(u0, x)
        arr = np.linspace(0, L, self.nx)
        der = np.zeros(nx)
        for i in range(self.nx):
            der[i] = du_dx.subs(x, arr[i])
        return der


class CheckpointingManager():
    """Manage the forward and backward solvers.

    Attributes
    ----------
    forward : object
        The forward solver.
    backward : object
        The backward solver.
    save_ram : int
        Number of checkpoint that will be stored.
    total_steps : int
        Total steps used to execute the solvers.

    """
    def __init__(self, max_n, equation, save_ram, save_disk):
        self.save_ram = save_ram
        self.save_disk = save_disk
        self.equation = equation
        self.max_n = max_n

    def execute(self):
        """Execute forward and adjoint with checkpointing mehtod.
        """
        @functools.singledispatch
        def action(cp_action):
            raise TypeError("Unexpected action")

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n
            u0 = self.equation.forward(u0, cp_action.n0, cp_action.n1)
            n1 = min(cp_action.n1, self.max_n)
            model_n = n1
            if cp_action.n1 == self.max_n:
                cp_schedule.finalize(n1)

        @action.register(Reverse)
        def action_reverse(cp_action):
            nonlocal model_r, p0
            nonlocal p0
            if model_r == 0:
                p0 = u
            p0 = self.equation(u0, p0)
            model_r += cp_action.n1 - cp_action.n0
            if cp_action.clear_adj_deps:
                data.clear()

        @action.register(Copy)
        def action_copy(cp_action):
            pass

        @action.register(EndForward)
        def action_end_forward(cp_action):
            pass

        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            pass

        model_n = 0
        model_r = 0
        p0 = None  # Initialiase the reverse computation.
        ics = set()
        data = set()

        snapshots = {StorageLocation(0).name: {}, StorageLocation(1).name: {}}
        cp_schedule = RevolveCheckpointSchedule(self.max_n, self.save_ram,
                                                snap_on_disk=self.save_disk)
        snapshots = {StorageLocation(0).name: {}, StorageLocation(1).name: {}}
        
        while True:
            cp_action = next(cp_schedule)
            action(cp_action)
            if isinstance(cp_action, EndReverse):  
                assert model_r == 0
                break

L = 1
nx = 500
dt = 0.01
T = 1.4
nu = 0.005

burger = burger_equation(L, nx, dt, T, nu)
x = np.linspace(0, L, nx)
# u0 =  np.sin(np.pi*x)
f_step = int(T/dt)
i_step = 0

t0 = np.exp(1/(8*nu))
e = np.exp(x*x/(4*nu))
t = 1
a = (t/t0)**(0.5)
u0 = (x/(t))/(1 + (a*e))

u_n = burger.forward(u0, i_step, f_step)
u = burger.forward_solution()
dudx = burger.dudx()
# Plot the final solution
plt.plot(x, u)
plt.plot(x, dudx)
plt.xlabel('x')
plt.ylabel('u')
# plt.title('')
plt.show()