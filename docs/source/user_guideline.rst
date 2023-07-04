.. _checkpoint_schedules-guide:
User guideline
==============

checkpoint_schedule application: adjoint-based gradient
-------------------------------------------------------

This user guideline describes the *checkpointing_schedules* package used
simplified case of adjoint-based gradient computation.

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
~~~~~~~~~~~~~~

We discretise both the forward and adjoint systems using the Finite
Element Method (FEM), following the discretisation that is explained in
detail in [1] that obtains the approximated solution by applying
Galerkin method with linear trial basis functions. The adjoint system is
also discretized with this same methodology.

Coding
~~~~~~

*BurgerGradAdj* provides the *forward* solver of Burger’s equation (2),
the *backaward* solver of the adjoint equation (4) and the computation
of the objective funtional (1). The constructor of the *BurgerGradAdj*
class defines the spatial and temporal configurations for the problem.
The forward equation system is implemented in the *forward* method,
whereas the adjoint equation system is implemented in the *backward*
method.

.. code:: ipython3

    from scipy.sparse import lil_matrix
    from scipy.optimize import newton_krylov
    import numpy as np
    from scipy.sparse.linalg import spsolve
    
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
            self.fwd_ic = u0
            self.lx = L
    
        def forward(self, u0, n0, n1, checkpointing=True):
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
            u = u0.copy()
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
            if not checkpointing:
                return u_sol
            else:
                return u_new
    
        def backward(self, u_fwd, p0, n0, n1, checkpointing=True):
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
            while t < steps:
                u[0] = u[nx - 1] = 0
                if checkpointing:
                    uf = u_fwd
                else:
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
    
      

Adjoint problem with *checkpoint_schedules* package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    import pickle
    class CheckpointingManager():
        """Manage the forward and backward solvers.
    
        Attributes
        ----------
        max_n : int
            Total steps used to execute the solvers.
        equation : object
            The object....
        backward : object
            The backward solver.
        save_ram : int
            Number of checkpoint that will be stored in RAM.
        save_disk : int
            Number of checkpoint that will be stored on disk.
        list_actions : list
            Store the list of actions.
        """
        def __init__(self, max_n, equation, save_ram, save_disk):
            self.max_n = max_n
            self.save_ram = save_ram
            self.save_disk = save_disk
            self.equation = equation
            self.list_actions = []
            
    
        def execute(self):
            """Execute forward and adjoint with checkpointing H-Revolve checkpointing method.
            """
            @functools.singledispatch
            def action(cp_action):
                raise TypeError("Unexpected action")
    
            @action.register(Forward)
            def action_forward(cp_action):
                nonlocal model_n, fwd_tape, ics, adj_deps
                if len(ics) == 0:
                    ics = {cp_action.n0: fwd_tape}
                    fwd_tape = None
    
                fwd_tape = self.equation.forward(ics[cp_action.n0], cp_action.n0, cp_action.n1)
    
                if cp_action.write_ics:
                    if cp_action.storage == StorageLocation(1).name:
                        file_name = "fwd_data/ufwd_"+ str(cp_action.n0) +".dat"
                        with open(file_name, "wb") as f:
                            pickle.dump(ics[cp_action.n0], f)
                        snapshots[cp_action.storage][cp_action.n0] = file_name
                    else:
                        snapshots[cp_action.storage][cp_action.n0] = ics[cp_action.n0]
                if cp_action.write_adj_deps:
                    adj_deps = {cp_action.n1: fwd_tape}
    
                ics.clear()
    
                n1 = min(cp_action.n1, self.max_n)
                model_n = n1
                if cp_action.n1 == self.max_n:
                    cp_schedule.finalize(n1)
    
            @action.register(Reverse)
            def action_reverse(cp_action):
                nonlocal model_r, bwd_tape, fwd_tape, adj_deps
                if model_r == 0:
                    # Initial condition of the adjoint system at the reverse step r=0.
                    p0 = fwd_tape
                    fwd_tape = None
                else:
                    # Initialise the adjoint system for the reverse step r > 0.
                    p0 = bwd_tape
    
                bwd_tape = self.equation.backward(adj_deps[cp_action.n1], p0, cp_action.n0, cp_action.n1)
                model_r += cp_action.n1 - cp_action.n0
                
                if cp_action.clear_adj_deps:
                    adj_deps.clear()
    
            @action.register(Copy)
            def action_copy(cp_action):
                nonlocal ics
                if cp_action.from_storage == StorageLocation(1).name:
                    file_name = snapshots[cp_action.from_storage][cp_action.n]
                    with open(file_name, "rb") as f:
                        data = np.asarray(pickle.load(f), dtype=float)
                else:
                    data = snapshots[cp_action.from_storage][cp_action.n]
    
                ics = {cp_action.n: data}
                if cp_action.delete:
                    del snapshots[cp_action.from_storage][cp_action.n]
    
            @action.register(EndForward)
            def action_end_forward(cp_action):
                pass
    
            @action.register(EndReverse)
            def action_end_reverse(cp_action):
                pass
    
            model_n = 0
            model_r = 0
            ics = {model_n: self.equation.fwd_ic}
            adj_deps = {}
            fwd_tape = None
            bwd_tape = None
    
            snapshots = {StorageLocation(0).name: {}, StorageLocation(1).name: {}}
            cp_schedule = RevolveCheckpointSchedule(self.max_n, self.save_ram,
                                                    snap_on_disk=self.save_disk)
            storage_limits = {StorageLocation(0).name: self.save_ram, 
                              StorageLocation(1).name: self.save_disk}
            if self.save_disk > 0 :
                import os 
                dir = "fwd_data"
                os.mkdir(dir)
            while True:
                cp_action = next(cp_schedule)
                action(cp_action)
                self.list_actions.append([str(cp_action)])
    
                # Checkpoint storage limits are not exceeded
                # for storage_type, storage_limit in storage_limits.items():
                #     assert len(snapshots[storage_type]) <= storage_limit
                    
                # Data storage limit is not exceeded
                assert min(1, len(ics)) + len(adj_deps) <= 1
                if isinstance(cp_action, EndReverse):  
                    x = np.linspace(0, self.equation.lx, self.equation.nx)
                    sens = np.trapz(bwd_tape*1.01*self.equation.fwd_ic, x=x, dx=self.equation.dx)
                    print(sens)
                    break
    


Firstly, let us consider few time-steps only to exemplify how it works
the forward and adjoint computations with *checkpoint_schedules*
package. So, we start by deffining the initial setup to execute an
adjoint problem with the employment of checkpointing method given buy
checkpoint_schedules\* package.

.. code:: ipython3

    L = 1  # Domain lenght
    nx = 500 # Number of nodes.
    nu = 0.005 # Viscosity
    dt = 0.01 # Time variation.
    T = 0.05 # Final time
    x = np.linspace(0, L, nx) 
    u0 = np.sin(np.pi*x)
    burger_grad_adj = BurgerGradAdj(L, nx, dt, T, nu, u0) # Defining the object...

Next, we want to get a manager object that is able to execute the
forward and adjoint equation by following the *checkpoint_schedules*
actions. To do that, we set the parameters necessary to obtain a
sequence of actions. They are the total time-steps, and the number of
checkpoint data that we want to store in RAM and on disk.

In this first example, we set checkpoint data associate to two steps of
the forward problem to be stored in RAM and one checkpoint data
associate to one step to be stored in disk.

.. code:: ipython3

    max_n = int(T/dt) # Total steps.
    save_ram = 2 # Number of steps to save in RAM.
    save_disk = 1 # Number of steps to save in disk.
    chk_manager = CheckpointingManager(max_n, burger_grad_adj, save_ram, save_disk)

After to define the manager object given by the *CheckpointingManager*
class, we execute our adjoint-based gradient problem by the
*chk_manager.execute()*.

.. code:: ipython3

    chk_manager.execute()


.. parsed-literal::

    11.975270553884258


To clarify how this adjoint problem works with the
*checkpoint_schedules* package, we have the list of actions used in this
first example given by the attribute *chk_manager.list_actions*.

.. code:: ipython3

    from tabulate import tabulate
    print(tabulate(chk_manager.list_actions, headers=["checkpoint_schedules actions"]))


.. parsed-literal::

    checkpoint_schedules actions
    -----------------------------------
    Forward(0, 3, True, False, 'RAM')
    Forward(3, 4, True, False, 'RAM')
    Forward(4, 5, False, True, 'RAM')
    EndForward()
    Reverse(5, 4, True)
    Copy(3, 'RAM', 'TAPE', True)
    Forward(3, 4, False, True, 'RAM')
    Reverse(4, 3, True)
    Copy(0, 'RAM', 'TAPE', False)
    Forward(0, 1, False, False, 'NONE')
    Forward(1, 2, True, False, 'RAM')
    Forward(2, 3, False, True, 'RAM')
    Reverse(3, 2, True)
    Copy(1, 'RAM', 'TAPE', True)
    Forward(1, 2, False, True, 'RAM')
    Reverse(2, 1, True)
    Copy(0, 'RAM', 'TAPE', True)
    Forward(0, 1, False, True, 'RAM')
    Reverse(1, 0, True)
    EndReverse(True,)


As we saw above, we have a list of *checkpoint_schedules* actions used
in the current adjoint problem. To untersdant them, let us remind the
actions in general form (this explanation is avaiable in the
introduction) and in the *checkpoint_schedules* API reference . \*
*Forward(n0, n1, write_ics, write_adj_deps, storage)*:

::

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

Therefore, for this particular case:

-  *Forward(0, 3, True, False, ‘RAM’)*:

   -  Execute the forward solver from step 0 to step 3.
   -  Write the forward data (*write_ics*) of step 0 to RAM (storage).
   -  The forward data is not stored for the adjoint computation
      (*write_adj_deps* is False).

-  *Forward(4, 5, False, True, ‘RAM’)*:

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

After to give an example over how the adjoint problem is executed, we
would like to clarify how the sequence of actions is created according
to the cost of save

References
~~~~~~~~~~

[1] Aksan, E. N. “A numerical solution of Burgers’ equation by finite
element method constructed on the method of discretization in time.”
Applied mathematics and computation 170.2 (2005): 895-904.

[2] Aupy, Guillaume, and Julien Herrmann. H-Revolve: a framework for
adjoint computation on synchrone hierarchical platforms.
(https://hal.inria.fr/hal-02080706/document), 2019.
