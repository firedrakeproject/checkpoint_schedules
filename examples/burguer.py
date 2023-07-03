import numpy as np
import functools
import matplotlib.pyplot as plt
from sympy import *
from scipy.sparse import lil_matrix

from scipy.optimize import newton_krylov
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

    def forward(self, u0, n0, n1):
        """
        Execute the forward system in time.

        Parameters
        ----------
        u0 : numpy.ndarray
            Forward initial condition.
        n0 : int
            Initial step.
        n1 : int
            Final step.

        Returns
        -------
        list
            List of solution arrays at each time step.
        """
        dx = self.dx
        nx = self.nx
        dt = self.dt
        nu = self.nu

        u = u0.copy()
        u_sol = [u]

        # Assemble the matrix system
        A = lil_matrix((nx, nx))
        B = lil_matrix((nx, nx))
        b = nu / (dx * dx)
        B[0, 0] = -1 / 3
        B[0, 1] = -1 / 6
        B[nx - 1, nx - 1] = -1 / 3
        B[nx - 1, nx - 2] = -1 / 6

        steps = int(n1 - n0)
        t = 0
        while t < steps:

            def non_linear(u_new):
                u[0] = u[nx - 1] = 0
                A[0, 0] = 1 / 3 - dt * (1/2*u_new[0] / dx + b)
                A[0, 1] = 1 / 6 + dt * (1 / 2 * u_new[0] / dx - b)
                A[nx - 1, nx - 1] = 1 / 3 - dt * (- u_new[nx - 1] / dx + b)
                A[nx - 1, nx - 2] = 1 / 6 + dt * (1 / 2 * u_new[nx - 2] / dx - b)

                for i in range(1, nx - 1):
                    B[i, i] = -2 / 3
                    B[i, i + 1] = B[i, i - 1] = -1 / 6
                    A[i, i - 1] = 1 / 6 - dt * (1 / 2 * u_new[i - 1] / dx + b)
                    A[i, i] = 2 / 3 + dt * (1 / 2 * (u_new[i - 1] - u_new[i]) / dx + 2 * b)
                    A[i, i + 1] = 1 / 6 + dt * (1 / 2 * u_new[i] / dx - b)

                F = A * u_new + B * u
                return F

            u_new = newton_krylov(non_linear, u)
            u = u_new.copy()
            u_sol.append(u_new)
            print(t)
            t += 1

        return u_sol

    
    def backward(self, u_fwd, p0, n0, n1):
        """Execute the adjoint system in time.

        Parameters
        ---------
        u_fwd : numpy array
            Forward solution that is the adjoint dependency.
        p0 : numpy array
            Adjoint solution used to initialize the adjoint solver.
        n0 : int
            Initial time step.
        n1 : int
            Final time step.
        """
        dx = self.dx
        nx = self.nx
        dt = self.dt
        b = self.nu / (dx * dx)
        u = p0.copy()
        u_new = np.zeros(nx)
        steps = int(n1 - n0)
        t = 0
        A = lil_matrix((nx, nx))
        B = lil_matrix((nx, nx))
        A[0, 0] = 1 / 3
        A[0, 1] = 1 / 6
        A[nx - 1, nx - 1] = 1 / 3
        A[nx - 1, nx - 2] = 1 / 6
        while t < steps - 1:
            u[0] = u[nx - 1] = 0
            uf = u_fwd[steps - 1 - t]
            B[0, 0] = 1 / 3 - dt * (uf[0] / dx - b - 1 / 3 * (uf[1] - uf[0]) / dx)
            B[0, 1] = 1 / 6 + dt * (1 / 2 * uf[0] / dx + b - 1 / 6 * (uf[2] - uf[1]) / dx)
            B[nx - 1, nx - 1] = 1 / 3 + dt * (uf[nx - 1] / dx - b - 1 / 3 * (uf[nx - 1] - uf[nx - 2]) / dx)
            B[nx - 1, nx - 2] = 1 / 6 + dt * (1 / 2 * u_new[nx - 2] / dx + b - 1 / 6 * (uf[nx - 1] - uf[nx - 2]) / dx)
            for i in range(1, nx - 1):
                v_m = uf[i] / dx
                v_mm1 = uf[i - 1] / dx
                deri = (uf[i] - uf[i - 1]) / dx
                derip = (uf[i + 1] - uf[i]) / dx
                A[i, i - 1] = 1 / 6
                A[i, i] = 2 / 3
                A[i, i + 1] = 1 / 6
                B[i, i] = 2 / 3 + dt * (1 / 2 * (v_mm1 - v_m) - 2 * b - 2 / 3 * (deri - derip))
                B[i, i - 1] = 1/6 - dt * (1 / 2 * v_mm1 - b - 1 / 6 * deri)
                B[i, i + 1] = 1/6 + dt*(1/2 * v_m + b - 1 / 6 * derip)
    
            d = B.dot(u)
            u_new = spsolve(A, d)
            u = u_new.copy()
            t += 1
        return u_new

    def forward_solution(self, t):
        u = np.zeros(self.nx)
        x = np.linspace(0, L, self.nx)
        t0 = np.exp(1/(8*self.nu))
        t = 1 + t
        for i in range(self.nx):
            e = np.exp(x[i]*x[i]/(4*self.nu*t))
            a = (t/t0)**(0.5)
            u[i] = (x[i]/(t))/(1 + (a*e))
        return u

    def dudx(self, t):
        x = Symbol('x')
        t0 = np.exp(1/(8*self.nu))
        e = exp(x*x/(4*self.nu))
        t = 1 + t
        a = (t/t0)**(0.5)
        u = (x/(t))/(1 + (a*e))
        dudx = diff(u, x)
        arr = np.linspace(0, L, self.nx)
        der = np.zeros(nx)
        for i in range(self.nx):
            der[i] = dudx.subs(x, arr[i])
        return der
    
    def num_energy(self, arr, x, dx):
        integ =np.trapz(arr, x=x, dx=dx)
        return integ

    def num_derivative(self, u):
        dudx = np.zeros(self.nx)
        dudx[0] = (u[1] - u[0])/self.dx
        for i in range(1, self.nx-1):
            deri = (u[i] - u[i-1])/self.dx
            derip1 = (u[i+1] - u[i])/self.dx
            dudx[i] = - 2*(derip1 - deri)
        return dudx
        
    def energy(self, t):
        re = 1/self.nu
        x = Symbol('x')
        t0 = np.exp(1/(8*self.nu))
        e = exp(x*x/(4*self.nu))
        t = 1 + t
        a = (t/t0)**(0.5)
        u = (x/(t))/(1 + (a*e))
        ene = u*0.5
        integ = integrate(ene, (x, 0, 1))
        
        print(integ)

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
nx = 100
dt = 0.001
T = 0.5
nu = 0.01
dx = L/nx
burger = burger_equation(L, nx, dt, T, nu)
x = np.linspace(0, L, nx)
# u0 =  np.sin(np.pi*x)
f_step = int(T/dt)
i_step = 0

# u_n0 = burger.forward(burger.forward_solution(0), i_step, f_step)
u_sol = []
der = []
steps = int(T/dt)

# burger.energy(T)
for s in range(steps):
    t = s*dt
    u = burger.forward_solution(t)
    # dudx = burger.dudx(t)
    # der.append(dudx)
    u_sol.append(u)
    # print(s)
u_num = burger.forward(burger.forward_solution(0), i_step, f_step)
u_num0 = burger.forward(1.01*burger.forward_solution(0), i_step, f_step)
# # num_der = burger.num_derivative(u_num)
integ = burger.num_energy(u_num[-1]*u_num[-1]*0.5, x, dx)
integ0 = burger.num_energy(u_num0[-1]*u_num0[-1]*0.5, x, dx)
p0 = u_num[-1]
p = burger.backward(u_num, p0, i_step, f_step)
sens = np.trapz(p*1.01*burger.forward_solution(0), x=x, dx=dx)
print(sens)
print((integ0 - integ)/0.01)



# 
print(p)
# Plot the final solution
plt.plot(x, p, label="adjoint")
plt.plot(x, u_num[-1], label="numerical")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
# plt.title('')
plt.show()