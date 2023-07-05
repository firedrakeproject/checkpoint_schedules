.. _checkpoint_schedules-guide:

User guideline
==============

*checkpoint_schedule* application: adjoint-based gradient
---------------------------------------------------------

This user guideline describes an adjoint-based gradient computation
using checkpointing given by *checkpointing_schedules* package.
Therefore, we initally define the adjoint-based gradient and then the
forward and adjoint solvers prescribed by *checkpointing_schedules*
package.

Defining the application
~~~~~~~~~~~~~~~~~~~~~~~~

Let us consider a one-dimensional (1D) problem where it aims to compute
the gradient/sensitivity of an objective functional :math:`I` with
respect to a control parameter. The objective functional, is given by
the expression:

.. math::


   I(u) = \int_{\Omega} \frac{1}{2} u(x, \tau)u(x, \tau) \, d x
   \tag{1}

which measures the energy of a 1D velocity variable :math:`u = u(x, t)`
governed by the 1D viscous Burgers equation, a nonlinear equation for
the advection and diffusion on momentum:

.. math::


   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0.
   \tag{2},

where :math:`x \in [0, L]` is the space variable and :math:`t`
represents the time variable. The boundary condition is
:math:`u(0, t) = u(L, t) = 0`, where :math:`L` is the lenght of the 1D
domain. The initial condition is given by
:math:`u(0, t) = u_0 = \sin(\pi x)`.

The goal is to compute the adjoint-based gradient of the objective
functional :math:`I(u)` with respect to the initial condition
:math:`u_0`. Hence, we need to define the adjoint system, which is a
function of the forward PDE (Partial Differential Equation) and the
objective functional of interest. The adjoint system can be obtained in
either a continuous or discrete space. In the continuous space approach,
the adjoint PDE is derived from the continuous forward PDE. On the other
hand, the discrete space approach utilizes the discrete forward PDE to
obtain a discrete adjoint system. Alternatively, a discrete adjoint can
be obtained through algorithmic differentiation (AD).

In this tutorial, the adjoint-based gradient is defined in the
continuous space. Thus, the adjoint-based gradient is given by the
expression:

.. math::


   \frac{\partial I}{\partial u_0} \delta u_0 = \int_{\Omega}  u^{\dagger}(x, 0) \delta u_0 \, dx,
   \tag{3}

where :math:`u^{\dagger}(x, 0)` is the adjoint variable governed by the
adjoint system:

.. math::


   -\frac{\partial u^{\dagger}}{\partial t} + u^{\dagger} \frac{\partial u}{\partial x} - u \frac{\partial u^{\dagger}}{\partial x} - \nu \frac{\partial^2 u^{\dagger}}{\partial x^2} = 0,
   \tag{4}

satisfying the boundary condition
:math:`u^{\dagger} (0, t) = u^{\dagger}(L, t) = 0`. In this case, the
initial condition is :math:`u^{\dagger} (x, \tau) = u(x, \tau)`.

The adjoint system time-reversed. Therefore, computing the adjoint-based
gradient requires storing the forward solution for each time-step, since
the adjoint equation depends on the forward solution as seen in adjoint
equation (4). Additionally, the gradient expression (3) is a function of
:math:`u^{\dagger} (0, t)`, which is the final adjoint time 0.

Discretisation
^^^^^^^^^^^^^^

Both the forward and adjoint systems are discretised using the Finite
Element Method (FEM), employing a discretisation methodology detailed in
[1]. This methodology uses the Galerkin method with linear trial basis
functions to obtain an approximate solution. The backward finite
difference method is employed to discretise the equations in time.

Coding
^^^^^^

The *BurgerGradAdj* class is implemented to set of functionalities for
solving Burger’s equation (2) and its corresponding adjoint equation
(4), as well as computing the objective functional (1). The
*BurgerGradAdj* class constructor is responsible for defining the
spatial and temporal configurations required for solving the problem. It
sets up the necessary parameters and initializes the problem domain.

.. code:: ipython3

    from scipy.sparse import lil_matrix
    from scipy.optimize import newton_krylov
    import numpy as np
    import pickle
    from scipy.sparse.linalg import spsolve
    import functools
    
    class BurgerGradAdj():
        """This class provides the solver of the non-linear forward burger's equation,
        the solver of the adjoint equation and the adjoint-based gradient computation.
    
        Attributes
        ----------
        lx : float
            The domain lenght.
        nx : int
            Number of nodes.
        dt : float
            Time step.
        T : float
            Final time.
        nu : float
            The viscosity.
        """
        def __init__(self, L, nx, dt, T, nu, u0):
            self.nt = int(T/dt)
            self.nu = nu
            self.dx = L / (nx - 1)
            self.dt = dt
            self.nx = nx
            self.u = {0: u0}
            self.lx = L
            self.snapshots = {'RAM': {}, 'DISK': {}}
            self.adj_deps = {}
            self.p = {}
        
      
        def forward(self, n0, n1, write_ics=False, write_adj_deps=False, storage=None, checkpointing=True):
            """Solve the non-linear forward burger's equation in time.
    
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
            numpy.ndarray|list
                Return the latest forward solution if the checkpointing method is employed, 
                else, return a list of the forward solution related to every time-step.
            """
            dx = self.dx
            nx = self.nx
            dt = self.dt
            nu = self.nu
            u = self.u[n0]
            if write_ics:
                if storage == 'RAM':
                    self.store_in_ram(u, n0)
                elif storage == 'DISK':
                    self.store_on_disk(u, n0)
    
            if not checkpointing:
                u_sol = []
                u_sol.append(u)
                
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
                if not checkpointing:
                    u_sol.append(u)
                t += 1
            
            if write_adj_deps:
                self.store_adj_deps(u_new, n1)
    
            self.update_fwd_initcondition(u_new, n1)
            if not checkpointing:
                return u_sol
            else:
                return u_new
    
        def backward(self, n0, n1, clear_adj_deps, checkpointing=True):
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
            u = self.p[n1]
            u_new = np.zeros(nx)
            steps = int(n1 - n0)
            t = 0
            A = lil_matrix((nx, nx))
            B = lil_matrix((nx, nx))
            A[0, 0] = 1 / 3
            A[0, 1] = 1 / 6
            A[nx - 1, nx - 1] = 1 / 3
            A[nx - 1, nx - 2] = 1 / 6
            while t < steps:
                u[0] = u[nx - 1] = 0
                if checkpointing:
                    uf = self.adj_deps[n1]
                else:
                    uf = self.adj_deps[steps - 1 - t]
    
                B[0, 0] = 1 / 3 - dt * (uf[0] / dx - b - 1 / 3 * (uf[1] - uf[0]) / dx)
                B[0, 1] = (1 / 6 + dt * (1 / 2 * uf[0] / dx + b - 1 / 6 * (uf[2] - uf[1]) / dx))
                B[nx - 1, nx - 1] = (1 / 3 + dt * (uf[nx - 1] / dx - b 
                                    - 1 / 3 * (uf[nx - 1] - uf[nx - 2]) / dx))
                B[nx - 1, nx - 2] = (1 / 6 + dt * (1 / 2 * u_new[nx - 2] / dx 
                                    + b - 1 / 6 * (uf[nx - 1] - uf[nx - 2]) / dx))
                
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
            self.update_bwd_initcondition(u_new, n0)
            if clear_adj_deps:
                self.adj_deps.clear()
            
    
        def copy_fwd_data(self, n, from_storage, delete):
            if from_storage == 'DISK':
                file_name = self.snapshots[from_storage][n]
                with open(file_name, "rb") as f:
                    u0 = np.asarray(pickle.load(f), dtype=float)
            else:
                u0 = self.snapshots[from_storage][n]
            self.update_fwd_initcondition(u0, n)  
            if delete:
                del self.snapshots[from_storage][n]
        
        def compute_grad(self):
            x = np.linspace(0, self.lx, self.nx)
            sens = np.trapz(self.p[0]*1.01*np.sin(np.pi*x), x=x, dx=self.dx)
            print("Sensitivity:", sens)
        
        def update_fwd_initcondition(self, data, n):
            self.u.clear()
            self.u = {n: data}
    
        def update_bwd_initcondition(self, data, n):
            self.p.clear()
            self.p = {n: data}
    
        def adj_initcondition(self, ic, n):
            self.p = {n: ic}
        
        def store_in_ram(self, data, step):
            """Store the forward data in RAM.
            """
            self.snapshots['RAM'][step] = data
    
        def store_on_disk(self, data, step):
            """Store the forward data on disk.
            """
            file_name = "fwd_data/ufwd_"+ str(step) +".dat"
            with open(file_name, "wb") as f:
                pickle.dump(data, f)
            self.snapshots['DISK'][step] = file_name
            
        def store_adj_deps(self, data, n):
            self.adj_deps = {n: data}
    
      

Using *checkpoint_schedules* package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*checkpoint_schedules* package provides a set of actions used to execute
the forward and adjoint solvers with the H-Revolve checkpointing method
[2]. Therefore, it is essential to import the actions (*Forward,
EndForward, Reverse, Copy, EndReverse*) to ensure proper functionality.
Also, *checkpoint_schedules* provides the checkpoint schedules iterator
*RevolveCheckpointSchedule*, where the actions in the schedule are
accessed by iterating over a sequence.

The actions are implemented using single-dispatch functions as carried
out in *CheckpointingManager* class, which provides a management of the
forward and adjoint solvers coordinated by the sequence of actions given
by the *checkpoint_schedules* package.

.. code:: ipython3

    from checkpoint_schedules import Forward, EndForward, Reverse, Copy, EndReverse
    from checkpoint_schedules import RevolveCheckpointSchedule, StorageLocation
    import functools
    
    
    class CheckpointingManager():
        """Manage the forward and backward solvers.
    
        Attributes
        ----------
        max_n : int
            Total steps used to execute the solvers.
        adj_grad_problem : object
            Adjoint-based gradient object.
        backward : object
            The backward solver.
        save_ram : int
            Number of checkpoint that will be stored in RAM.
        save_disk : int
            Number of checkpoint that will be stored on disk.
        list_actions : list
            Store the list of actions.
        """
        def __init__(self, max_n, adj_grad_problem, save_ram, save_disk):
            self.max_n = max_n
            self.save_ram = save_ram
            self.save_disk = save_disk
            self.adj_grad_problem = adj_grad_problem
            self.list_actions = []
            
        def execute(self, cp_schedule):
            """Execute forward and adjoint with checkpointing H-Revolve checkpointing method.
            """
            @functools.singledispatch
            def action(cp_action):
                raise TypeError("Unexpected action")
    
            @action.register(Forward)
            def action_forward(cp_action):
                self.adj_grad_problem.forward(cp_action.n0, cp_action.n1, 
                                      write_ics=cp_action.write_ics, 
                                      write_adj_deps=cp_action.write_adj_deps,
                                      storage=cp_action.storage)
    
                n1 = min(cp_action.n1, self.max_n)
                if cp_action.n1 == self.max_n:
                    cp_schedule.finalize(n1)
    
            @action.register(Reverse)
            def action_reverse(cp_action):
                nonlocal model_r
                self.adj_grad_problem.backward(cp_action.n0, cp_action.n1, 
                                               clear_adj_deps=cp_action.clear_adj_deps)
                model_r += cp_action.n1 - cp_action.n0
                
            @action.register(Copy)
            def action_copy(cp_action):
                self.adj_grad_problem.copy_fwd_data(cp_action.n, cp_action.from_storage, cp_action.delete)
    
            @action.register(EndForward)
            def action_end_forward(cp_action):
                ic = self.adj_grad_problem.u
                self.adj_grad_problem.adj_initcondition(ic[self.max_n], self.max_n)
    
            @action.register(EndReverse)
            def action_end_reverse(cp_action):
                self.adj_grad_problem.compute_grad()
    
            model_n = 0
            model_r = 0
    
            storage_limits = {StorageLocation(0).name: self.save_ram, 
                              StorageLocation(1).name: self.save_disk}
    
            count = 0
            while True:
                cp_action = next(cp_schedule)
                action(cp_action)
                self.list_actions.append([count, str(cp_action)])
                count += 1
                if isinstance(cp_action, EndReverse):  
                    break
    


Let us consider few time-steps only to exemplify how it works the
forward and adjoint computations with *checkpoint_schedules* package.
So, we start by deffining the basic problem setup.

.. code:: ipython3

    L = 1  # Domain lenght
    nx = 500 # Number of nodes.
    nu = 0.005 # Viscosity
    dt = 0.01 # Time variation.
    T = 0.05 # Final time
    x = np.linspace(0, L, nx) 
    u0 = np.sin(np.pi*x)
    burger_grad_adj = BurgerGradAdj(L, nx, dt, T, nu, u0) # Defining the object able to execute forward/adjoint solvers and the computation of the cost function.

We want to get a manager object able to execute the forward and adjoint
equations by following the *checkpoint_schedules* actions. To do that,
we set the parameters necessary to obtain a sequence of actions. They
are the total time-steps, and the number of checkpoint data that we want
to store in RAM and on disk.

In this first example, we set checkpoint data associate to two steps of
the forward problem to be stored in RAM and one checkpoint data
associate to one step to be stored in disk.

.. code:: ipython3

    max_n = int(T/dt) # Total steps.
    save_ram = 2 # Number of steps to save in RAM.
    save_disk = 0 # Number of steps to save in disk.
    chk_manager = CheckpointingManager(max_n, burger_grad_adj, save_ram, save_disk) # manager object able to execute the forward and adjoint equations

After to define the manager object given by the *CheckpointingManager*
class, we execute our adjoint-based gradient problem by the ``execute``
method as shown below, where the execution depends of the checkpoint
schedule that is built from a list of checkpoint operations provided by
the H-Revolve checkpointing method.

.. code:: ipython3

    cp_schedule = RevolveCheckpointSchedule(max_n, save_ram, snap_on_disk=save_disk)
    chk_manager.execute(cp_schedule)


.. parsed-literal::

    /Users/ddolci/work/checkpoint_schedules/.venv/lib/python3.11/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py:214: SparseEfficiencyWarning: spsolve requires A be CSC or CSR matrix format
      warn('spsolve requires A be CSC or CSR matrix format',


.. parsed-literal::

    Sensitivity: 12.001369298736885


To clarify how this adjoint problem works with the
*checkpoint_schedules* package, we have the list of actions used in this
first example given by the attribute ``chk_manager.list_actions``.

.. code:: ipython3

    from tabulate import tabulate
    print(tabulate(chk_manager.list_actions, headers=["Action number", "checkpoint_schedules actions"]))


.. parsed-literal::

      Action number  checkpoint_schedules actions
    ---------------  -----------------------------------
                  0  Forward(0, 3, True, False, 'RAM')
                  1  Forward(3, 4, True, False, 'RAM')
                  2  Forward(4, 5, False, True, 'RAM')
                  3  EndForward()
                  4  Reverse(5, 4, True)
                  5  Copy(3, 'RAM', 'TAPE', True)
                  6  Forward(3, 4, False, True, 'RAM')
                  7  Reverse(4, 3, True)
                  8  Copy(0, 'RAM', 'TAPE', False)
                  9  Forward(0, 1, False, False, 'NONE')
                 10  Forward(1, 2, True, False, 'RAM')
                 11  Forward(2, 3, False, True, 'RAM')
                 12  Reverse(3, 2, True)
                 13  Copy(1, 'RAM', 'TAPE', True)
                 14  Forward(1, 2, False, True, 'RAM')
                 15  Reverse(2, 1, True)
                 16  Copy(0, 'RAM', 'TAPE', True)
                 17  Forward(0, 1, False, True, 'RAM')
                 18  Reverse(1, 0, True)
                 19  EndReverse(True,)


As we saw above, we have a list of *checkpoint_schedules* actions used
in the current adjoint problem. To untersdant them, let us remind the
actions in general form (this explanation is avaiable in the
introduction and in the *checkpoint_schedules* API reference): 

- *Forward(n0, n1, write_ics, write_adj_deps, storage)*:
   - Executes the forward solver from step *n0* to step *n1*.
   - Write the forward data of step *n0* if *write_ics* is *True*.
   - Indicates whether to store the forward data for the adjoint computation (*write_adj_deps*).
   - Indicate the storage level for the forward data (storage).

-  *Reverse(n0, n1, clear_adj_deps)*:

   -  Executes the adjoint solver from step *n0* to step *n1*.
   -  Clears the adjoint dependencies (*adj_deps*) used in the adjoint
      computation.

-  *Copy(n, from_storage, to_storage, delete)*:

   -  Copy the forward data related to step n from one storage location
      (*from_storage*) to another storage location (*to_storage*).
   -  Indicate whether to delete the copied data from the source storage
      location (delete).

-  *EndForward()*:

   -  Indicates the finalization of the forward solver.

-  *EndReverse()*:

   -  Indicate the finalisation of the adjoint solver.

Therefore, for the currrent particular case we have some explanations
relations to some actions:

-  Action number 0: *Forward(0, 3, True, False, ‘RAM’)*:

   -  Execute the forward solver from step 0 to step 3.
   -  Write the forward data (*write_ics*) of step 0 to RAM (storage).
   -  The forward data is not stored for the adjoint computation
      (*write_adj_deps* is False).

-  Action number 1: *Forward(4, 5, False, True, ‘RAM’)*:

   -  Execute the forward solver from step 4 to step 5.
   -  Do not write the forward data (*write_ics*) of step 4.
   -  Store the forward data for the adjoint computation
      (*write_adj_deps* is *True*) of step 5 in RAM.

-  *Reverse(4, 3, True)*:

   -  Execute the adjoint solver from step 4 to step 3.
   -  Clear the adjoint dependencies (*adj_deps*) used in the adjoint
      computation.

-  Copy(0, ‘RAM’, ‘TAPE’, False):

   -  Copy the forward data related to step 0 from RAM to TAPE.
   -  Do not delete the copied data from RAM (*delete* is *False*) since
      it will be used again to restart the forward solver.

-  Copy(0, ‘RAM’, ‘TAPE’, True):

   -  Copy the forward data related to step 0 from RAM to TAPE.
   -  Delete the copied data from RAM (*delete* is *True*) as it is not
      needed anymore.



References
~~~~~~~~~~

[1] Aksan, E. N. “A numerical solution of Burgers’ equation by finite
element method constructed on the method of discretization in time.”
Applied mathematics and computation 170.2 (2005): 895-904.

[2] Aupy, Guillaume, and Julien Herrmann. H-Revolve: a framework for
adjoint computation on synchrone hierarchical platforms.
(https://hal.inria.fr/hal-02080706/document), 2019.
