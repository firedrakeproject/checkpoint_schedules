.. _example_checkpoint_schedules:

Using *checkpoint_schedules*
============================

This example aims introduce the usage of *checkpoint_schedules* through
an initial illustration of how this package works and how to understand
the execution in time of forward and adjoint solvers using a schedule
and generators provided by *checkpointing_schedules* package.

Managing the forward and adjoint executions with schedules
----------------------------------------------------------

Let us consider the ``CheckpointingManager`` class, which plays the
essential role in managing forward and adjoint executions. This
management is given by iterating over a schedule with the execution of
``CheckpointingManager.execute(cp_schedule)``. To perform this task, the
method requires a ``cp_schedule`` argument, which must be a
*checkpoint_schedules* object containing a schedule attribute and a
generator method. This generator is responsible for yielding the
*checkpoint_schedules* actions to be executed.

Whitin ``CheckpointingManager.execute``, we implement the
*checkpointing_schedules* actions using single-dispatch functions. The
base function, ``action``, is decorated with the ``singledispatch``.
Specific actions functions are then established by using the register
method of the ``action``. Hence, ``CheckpointingManager.execute``
effectively calls the specific action according to a schedule.

.. code:: ipython3

    from checkpoint_schedules import *
    import functools
    
    class CheckpointingManager():
        """Manage the executions of the forward and adjoint solvers.
    
        Attributes
        ----------
        max_n : int
            Total steps used to execute the solvers.
        """
        def __init__(self, max_n):
            self.max_n = max_n
            self.list_actions = []
            self.index = 0
            
        def execute(self, cp_schedule):
            """Execute forward and adjoint with a checkpointing schedule.
    
            Parameters
            ----------
            cp_schedule : CheckpointSchedule
                Checkpoint schedule object.
            """
            @functools.singledispatch
            def action(cp_action):
                raise TypeError("Unexpected action")
    
            @action.register(Forward)
            def action_forward(cp_action):
                
                nonlocal model_n
                n1 = min(cp_action.n1, self.max_n)
                .. execute_forward_solver(cp_action.n0, n1)
                # forward time steps
                model_n = n1
                if n1 == self.max_n:
                    # Imposing the latest time step.
                    cp_schedule.finalize(n1)
    
            @action.register(Reverse)
            def action_reverse(cp_action):
                nonlocal model_r
                .. execute_adjoint_solver(cp_action.n0, cp_action.n1)
                # reverse time steps
                model_r += cp_action.n1 - cp_action.n0
                
            @action.register(Copy)
            def action_copy(cp_action):
                .. copy_forward_data(cp_action.n)
    
            @action.register(Move)
            def action_move(cp_action):
                .. move_forward_data(cp_action.n)
    
            @action.register(EndForward)
            def action_end_forward(cp_action):
                assert model_n == self.max_n
                if cp_schedule._max_n is None:
                    cp_schedule._max_n = self.max_n
                
            @action.register(EndReverse)
            def action_end_reverse(cp_action):
                nonlocal model_r, is_exhausted
                assert model_r == self.max_n
                is_exhausted = cp_schedule.is_exhausted
                
            model_n = 0
            model_r = 0
            is_exhausted = False # indicates if the schedule is exhausted
    
            for count, cp_action in enumerate(cp_schedule):
                self.index = count
                action(cp_action)
                if isinstance (cp_action, EndReverse) and is_exhausted is False:
                    # In the cases of online checkpoints, the adjoint execution 
                    # can be executed multiple times. That is undesirable within 
                    # this specific illustration. Therefore, the adjoint process 
                    # is here limited to a single execution.              
                    break
                
            from tabulate import tabulate
            print(tabulate(self.list_actions, headers=['Action index:', 'Run-time illustration', 
                                                        'Action:']))
            self.list_actions = []


Schedule for no adjoint computation
-----------------------------------

Firstly, let us define the maximum solvers time steps ``max_n = 4`` to make easier the visual
representation of the execution.
Next, we instantiate an object named ``solver_manager`` of the
``CheckpointingManager`` class, using the ``max_n`` as an argument.

.. code:: ipython3

    max_n = 4 # Total number of time steps.
    solver_manager = CheckpointingManager(max_n) # manager object

The :class:`NoneCheckpointSchedule` class provides a schedule object that is
a parameter within the ``solver_manager.execute`` method. In this case, 
the built schedule executes only the forward solver, excluding any data storage.

.. code:: ipython3

    cp_schedule = NoneCheckpointSchedule() # Checkpoint schedule object
    solver_manager.execute(cp_schedule) # Execute the forward solver by following the schedule.


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------------------
                  0  −−−▷−−−▷−−−▷−−−▷         Forward(0, 9223372036854775807, False, False, <StorageType.NONE: None>)
                  1  End Forward              EndForward()


The output above brings informations when a solver is executed with a schedule  given by :class:`NoneCheckpointSchedule`.
Notice in the output that we have two actions: :class:`Forward` and :class:`EndForward`. 
The fundamental structure of the :class:`Forward` action is given by:

.. code:: python

   Forward(n0, n1, write_ics, write_adj_deps, storage_type)

This action is read as: 
- Advance the forward solver from step ``n0`` to the start of any step ``n1``. 
- ``write_ics`` and ``write_adj_deps`` are booleans that indicate whether the 
forward solver should store the forward restart data and the forward data required for the adjoint
computation, respectively. 
- ``storage_type`` is an enum that indicates the type of storage required for the forward restart 
data and the forward data required for the adjoint computation.

Within the context of the :class:`NoneCheckpointSchedule` schedule, the
:class:`Forward` action is then given by:

.. code:: python

   Forward(0, 9223372036854775807, False, False, <StorageType.NONE: None>)

-  Advance the forward solver from step ``n0 = 0`` to the start of any
   step ``n1``.
-  Both ``write_ics`` and ``write_adj_deps`` are set to ``'False'``,
   indicating that the forward solver does not store the forward restart
   data and the forward data required for the adjoint computation.
-  The storage type is ``StorageType.NONE``, indicating that no specific
   storage type is required.

* This schedule is built without specifying a maximum step for the
forward solver execution. That is the reason to obtain this immeasurable
number for ``n1``. Therefore, using the :class:`NoneCheckpointSchedule`
schedule offers the flexibility to determine the desired steps while the
forward solver is time advancing.*

In the current example, we determine the maximum step ``max_n = 4``, an
attribute within the ``CheckpointingManager``. Next, we conclude the
forward solver execution with the following python script:

.. code:: python

    cp_schedule.finalize(n1)

where ``n1 = max_n = 4``. This line is incorporated in the
``action_forward`` that is ``singledispatch`` registered function in
``CheckpointingManager.execute``.

Another action provided by the current schedule is the ``EndForward()``,
which indicates the forward solver has reached the end of the time
interval.

Schedule for storing all time step forward data
-----------------------------------------------

We now begin to present the schedules when the adjoint solver is considered.

The following code is valuable for the cases where the user intend to
store the forward data for all time steps and does not applies any
checkpointing strategy.

The schedule in which there is the storage of the forward data for all
time steps in ``'RAM'`` is achieved by using the
:class:`SingleMemoryStorageSchedule` class. The code below shows the employment 
of this schedule in both forward and adjoint computations.

*Storing the forward restart data is unnecessary by this schedule, as
there is no need to recompute the forward solver while time advancing
the adjoint solver.*

*The :class:`SingleMemoryStorageSchedule` schedule offers the flexibility to
determine the desired steps while the forward solver is time advancing. *

.. code:: ipython3

    cp_schedule = SingleMemoryStorageSchedule()
    solver_manager.execute(cp_schedule)



.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------------------
                  0  −−−▶−−−▶−−−▶−−−▶         Forward(0, 9223372036854775807, False, True, <StorageType.ADJ_DEPS: 3>)
                  1  End Forward              EndForward()
                  2  ◀−−−◀−−−◀−−−◀−−−         Reverse(4, 0, True)
                  3  End Reverse              EndReverse(False,)


In this particular case, the :class:`Forward` action is given by:

.. code:: python

   Forward(0, 9223372036854775807, False, True, <StorageType.RAM: 0>)

which reads 
- Advance the forward solver from the step ``n0=0`` to the
start of any step ``n1``. 
- Do not store the forward restart data once
if ``write_ics`` is ``'False'``. 
- Store the forward data required for
the adjoint computation once ``write_adj_deps`` is ``'True'``. 
- Storage type is ``<StorageType.ADJ_DEPS: 3>``, which indicates the storage in a
``'local'`` that holds the forward data required for the adjoint
computation.

When the adjoint computation is considered in the schedule, we have the
*Reverse* action that fundamentally is given by:

.. code:: python

   Reverse(n0, n1, clear_adj_deps)

This is interpreted as follows: 
- Advance the adjoint solver from the step ``n0`` to the start of the step ``n1``. 
- Clear the adjoint dependency data if ``clear_adj_deps`` is ``'True'``.

In the current context, the *Reverse* action is:

.. code:: python

   Reverse(4, 0, True)

-  Advance the forward solver from the step ``4`` to the start of the
   step ``0``.
-  Clear the adjoint dependency (forward data) once ``clear_adj_deps``
   is ``'True'``.

When adjoint computations are taken into account in the schedules, the 
``EndReverse(True)`` action is required to indicate the end of the adjoint 
advancing.

The *checkpoint_schedules* additionally allows users to execute forward
and adjoint solvers while storing all adjoint dependencies on
``'disk'``. The following code shows this schedule applied in the
forward and adjoint executions with the object generated by the
:class:`SingleDiskStorageSchedule` class.

.. code:: ipython3

    cp_schedule = SingleDiskStorageSchedule()
    solver_manager.execute(cp_schedule)



.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -------------------------------------------------------------------
                  0  −−−▶−−−▶−−−▶−−−▶         Forward(0, 9223372036854775807, False, True, <StorageType.DISK: 1>)
                  1  End Forward              EndForward()
                  2                           Copy(4, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Copy(3, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  5  .   .   ◀−−−             Reverse(3, 2, True)
                  6                           Copy(2, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  7  .   ◀−−−                 Reverse(2, 1, True)
                  8                           Copy(1, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  9  ◀−−−                     Reverse(1, 0, True)


In the case illustrated above, forward and adjoint executions with
:class:`SingleDiskStorageSchedule` also have the *Copy* action (see the
outputs associated with the indexes 2, 4, 6, 8) which indicates copying
of the forward data from one storage type to another.

The *Copy* action has the fundamental structure:

.. code:: python

   Copy(n, from_storage, to_storage)

which reads: 
- Copy the data associated with step ``n``. 
- The term ``from_storage`` denotes the storage type responsible for retaining
forward data at step n, while ``to_storage`` refers to the designated
storage type for storing this forward data.

Hence, on considering the *Copy* action associated with the output
``Action index 4``, we have:

.. code:: python

   Copy(4, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)

This action reads: 
- Copy the data associated with step ``4``.
 - The forward data is copied from ``'disk'`` storage, and the specified
storage type to copy (``StorageType.ADJ_DEPS``) refers to the
``'local'`` storage that retains the forward data essential for the
adjoint computation.

Now, let us consider the case where the objective is to move the data
from one storage type to another insteady of copying it. To achieve
this, the optional ``move_data`` parameter within the
:class:`SingleDiskStorageSchedule` need to be set as ``True``. This
configuration is illustrated in the following code example:

.. code:: ipython3

    cp_schedule = SingleDiskStorageSchedule(move_data=True)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -------------------------------------------------------------------
                  0  −−−▶−−−▶−−−▶−−−▶         Forward(0, 9223372036854775807, False, True, <StorageType.DISK: 1>)
                  1  End Forward              EndForward()
                  2                           Move(4, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Move(3, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  5  .   .   ◀−−−             Reverse(3, 2, True)
                  6                           Move(2, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  7  .   ◀−−−                 Reverse(2, 1, True)
                  8                           Move(1, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                  9  ◀−−−                     Reverse(1, 0, True)


The *Move* action follows a basic structure:

.. code:: python

   Move(n, from_storage, to_storage)

This can be understood as: 
- Move the data associated with step ``n``. 
- The terms ``from_storage`` and ``to_storage``
hold the same meaning as in the *Copy* action.

Now, on considering one of the *Move* action associated with the output
``Action index: 4``:

.. code:: python

   Move(4, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)

Interpreted as: 
- Move the data associated with the step ``4``.
- The forward data is moved from ``'disk'`` storage to a storage used for the
adjoint computation.

**The Move action entails that the data, once moved, becomes no longer
accessible in the original storage type. Whereas the Copy action means
that the copied data remains available in the original storage type.**

Schedules given by checkointing methods
---------------------------------------

Revolve
~~~~~~~

Now, let us consider the schedules given by the checkpointing
strategies. We begin by employing the Revolve approach, according to
introduced in reference [1].

The Revolve checkpointing strategy generates a schedule that only uses
``'RAM'`` storage.

The :class::`Revolve` class gives a schedule according to two essential
parameters: the total count of forward time steps (``max_n = 4``) and
the number of checkpoints to store in ``'RAM'`` (``snaps_in_ram = 2``).

The code below represents the execution of the forward and adjoint solvers
with the the :class::`Revolve` schedule.

.. code:: ipython3

    snaps_in_ram = 2 
    solver_manager = CheckpointingManager(max_n) # manager object
    cp_schedule = Revolve(max_n, snaps_in_ram) 
    solver_manager.execute(cp_schedule)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------
                  0  +−−−▷−−−▷                Forward(0, 2, True, False, <StorageType.RAM: 0>)
                  1  .   .   +−−−▷            Forward(2, 3, True, False, <StorageType.RAM: 0>)
                  2  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  3  End Forward              EndForward()
                  4  .   .   .   ◀−−−         Reverse(4, 3, True)
                  5                           Move(2, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                 13  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 14  ◀−−−                     Reverse(1, 0, True)
                 15  End Reverse              EndReverse()


The employment of the checkpointing strategies within an adjoint-based
gradient requires the forward solver restart and recomputation.

The :class:`Forward` action associated with the ``Action index: 0`` that is read as follows:

.. code:: python

   Forward(0, 2, True, False, <StorageType.RAM: 0>)

-  Advance from time step 0 to the start of the time step 2.
-  Store the forward data required to restart the forward solver from
   time step 0.
-  The storage of the forward restart data is done in RAM.

* In the displayed time step illustrations, we have ``'+−−−▷−−−▷'`` associated to

.. code:: python

   Forward(0, 2, True, False, <StorageType.RAM: 0>)

The symbol ``'+'`` indicates that the forward data necessary for
restarting the forward computation from step 0 is stored. In the time
illustrations, the illustration ``'−−−▷'`` indicates that the forward
data used for the adjoint computation is **not** stored. When you
encounter ``'−−−▶'``, one indicates that the forward data is stored.

To summarize:

- ``'+'``: Forward data for restarting is stored. 
- ``'−−−▷'``: Forward data for adjoint computation is not stored. 
- ``'−−−▶'``: Forward data is stored for adjoint computation.

Multistage checkpoiting
~~~~~~~~~~~~~~~~~~~~~~~

The schedule as depicted below, employes a *MultiStage* distribution of
checkpoints between ``'RAM'`` and ``'disk'`` as described in [2]. This
checkpointing allows exclusively the memory storage (``'RAM'``), or
exclusively the ``'disk'`` storage, or in both storage locations.

The following code use two types of storage, ``'RAM'`` and ``'disk'``.

*MultiStage* checkpointing schedule is given by
:class:`MultistageCheckpointSchedule`, which requires the parameters: number
of checkpoints stored in ``'RAM'`` and ``'disk'``.

See below the code and output illustrating the forward and adjoint executions with
:class:`MultistageCheckpointSchedule`.

.. code:: ipython3

    snaps_in_ram = 1  # number of checkpoints stored in RAM
    snaps_on_disk = 1 # number of checkpoints stored in disk
    cp_schedule = MultistageCheckpointSchedule(max_n, snaps_in_ram, snaps_on_disk)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  ------------------------------------------------------------
                  0  +−−−▷−−−▷                Forward(0, 2, True, False, <StorageType.RAM: 0>)
                  1  .   .   +−−−▷            Forward(2, 3, True, False, <StorageType.DISK: 1>)
                  2  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  3  End Forward              EndForward()
                  4  .   .   .   ◀−−−         Reverse(4, 3, True)
                  5                           Move(2, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                 13  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 14  ◀−−−                     Reverse(1, 0, True)
                 15  End Reverse              EndReverse()


Disk-Revolve
~~~~~~~~~~~~

The following code shows the the execution of a solver over time using
the *Disk-Revolve* schedule, as described in reference [3]. This
schedule considers two type of storage: memory (``'RAM'``) and
``'disk'``.

The *Disk-Revolve* algorithm, available within the
*checkpoint_schedules* is given by :class:`DiskRevolve` that requires 
the definition of checkpoints stored in
memory to be greater than 0 (``'snap_in_ram > 0'``). Specifying the
checkpoints stored on ``'disk'`` is not required, as the algorithm
itself calculates this value.

The number of checkpoints stored in ``'disk'`` is determined according
the costs associated with advancing the backward and forward solvers in
a single time step, and the costs of writing and reading the checkpoints
saved on disk. Additional details of the definition of these parameters
can be found in the references [3], [4] and [5].

.. code:: ipython3

    snaps_in_ram = 1 # number of checkpoints stored in RAM
    cp_schedule = DiskRevolve(max_n, snapshots_in_ram=snaps_in_ram) # checkpointing schedule object
    solver_manager.execute(cp_schedule)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------
                  0  +−−−▷−−−▷−−−▷            Forward(0, 3, True, False, <StorageType.RAM: 0>)
                  1  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  2  End Forward              EndForward()
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  5  −−−▷−−−▷                 Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                 13  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 14  ◀−−−                     Reverse(1, 0, True)
                 15  End Reverse              EndReverse()


Periodic Disk Revolve
~~~~~~~~~~~~~~~~~~~~~

The schedule employed in the following code was presented in reference [4].
It is referred to as *Periodic Disk Revolve*. 

Analogously to the *Disk Revolve* schedule, one requires the specification of the maximum number of steps
(``max_n``) and the number of checkpoints saved in memory
(``snaps_in_ram``). The *Periodic Disk Revolve* also computes automatically
the number of checkpoint stored in disk. 

The class :class:`PeriodicDiskRevolve` provides the schedule object for this algorithm. *It is essential 
for the number of checkpoints in ``'RAM'`` to be greater than zero (``'snap_in_ram > 0'``)*

.. code:: ipython3

    snaps_in_ram = 1
    cp_schedule = PeriodicDiskRevolve(max_n, snaps_in_ram)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    We use periods of size  3
      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------
                  0  +−−−▷−−−▷−−−▷            Forward(0, 3, True, False, <StorageType.RAM: 0>)
                  1  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  2  End Forward              EndForward()
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  5  −−−▷−−−▷                 Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                 13  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 14  ◀−−−                     Reverse(1, 0, True)
                 15  End Reverse              EndReverse()


H-Revolve
~~~~~~~~~

The following code illustrates the forward and adjoint computations
using the checkpointing given by H-Revolve strategy [5]. This
checkpointing schedule is generated with :class: `HRevolve` class, which
requires the following parameters: maximum steps stored in RAM
(``snap_in_ram``), maximum steps stored on disk (``snap_on_disk``), and
the number of time steps (``max_n``).

*It is essential for the number of checkpoints in ``'RAM'`` to be
greater than zero (``'snap_in_ram > 0'``)*

.. code:: ipython3

    snaps_on_disk = 1
    snaps_in_ram = 1
    cp_schedule = HRevolve(max_n, snaps_in_ram, snaps_on_disk)  # checkpointing schedule
    solver_manager.execute(cp_schedule) # execute forward and adjoint in time with the schedule


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------
                  0  +−−−▷−−−▷−−−▷            Forward(0, 3, True, False, <StorageType.RAM: 0>)
                  1  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  2  End Forward              EndForward()
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  5  −−−▷−−−▷                 Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                 13  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 14  ◀−−−                     Reverse(1, 0, True)
                 15  End Reverse              EndReverse()


Mixed checkpointing
~~~~~~~~~~~~~~~~~~~

The *Mixed* checkpointing strategy works under the assumption that the
data required to restart the forward computation is of the same size as
the data required to advance the adjoint computation in one step.
Further details into the *Mixed* checkpointing schedule was discussed in
reference [6].

This specific schedule provides the flexibility to store the forward
restart data either in ``'RAM'`` or on ``'disk'``, but not both
simultaneously within the same schedule. The following code illustrates
the forward and adjoint computations using the Mixed checkpointing reached
with :class: `MixedCheckpointSchedule` class.

.. code:: ipython3

    snaps_on_disk = 1
    max_n = 4
    cp_schedule = MixedCheckpointSchedule(max_n, snaps_on_disk)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  ------------------------------------------------------------
                  0  +−−−▷−−−▷−−−▷            Forward(0, 3, True, False, <StorageType.DISK: 1>)
                  1  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  2  End Forward              EndForward()
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  5  −−−▷−−−▷                 Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Move(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▶                     Forward(0, 1, False, True, <StorageType.DISK: 1>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
                 13  ◀−−−                     Reverse(1, 0, True)
                 14  End Reverse              EndReverse()


In the example mentioned earlier, the storage of the forward restart
data is default configured for ``'disk'``. To modify the storage type to
``'RAM'``, the user can set the :class:`MixedCheckpointSchedule` argument
``storage = StorageType.RAM``, as displayed below.

.. code:: ipython3

    snaps_in_ram = 1
    cp_schedule = MixedCheckpointSchedule(max_n, snaps_on_disk, storage=StorageType.RAM)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  -----------------------------------------------------------
                  0  +−−−▷−−−▷−−−▷            Forward(0, 3, True, False, <StorageType.RAM: 0>)
                  1  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  2  End Forward              EndForward()
                  3  .   .   .   ◀−−−         Reverse(4, 3, True)
                  4                           Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  5  −−−▷−−−▷                 Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
                  6  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  7  .   .   ◀−−−             Reverse(3, 2, True)
                  8                           Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                  9  −−−▶                     Forward(0, 1, False, True, <StorageType.RAM: 0>)
                 10  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 11  .   ◀−−−                 Reverse(2, 1, True)
                 12                           Move(0, <StorageType.RAM: 0>, <StorageType.ADJ_DEPS: 3>)
                 13  ◀−−−                     Reverse(1, 0, True)
                 14  End Reverse              EndReverse()


Two-level binomial
~~~~~~~~~~~~~~~~~~

Two-level binomial schedule was presented in reference [6], and its
application was performed in the work [7].

The two-level binomial checkpointing stores the forward restart data
based on the user-defined ``period``. In this schedule, the user also
define the limite for additional storage of the forward restart data to
use during the advancing of the adjoint between periodic storage
checkpoints. The default sotrage type is ``'disk'``.

Now, let us define the period of storage ``period = 2`` and the extra
forward restart data storage ``add_snaps = 1``. The code displayed below
shows the execution in time illustration for this setup, where the schedule
object is given by the :class:`TwoLevelCheckpointSchedule` class.

.. code:: ipython3

    add_snaps = 1 # of additional storage of the forward restart data
    period = 2
    revolver = TwoLevelCheckpointSchedule(period, add_snaps)
    solver_manager.execute(revolver)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  ------------------------------------------------------------
                  0  +−−−▷−−−▷                Forward(0, 2, True, False, <StorageType.DISK: 1>)
                  1  .   .   +−−−▷−−−▷        Forward(2, 4, True, False, <StorageType.DISK: 1>)
                  2  End Forward              EndForward()
                  3                           Copy(2, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  4  .   .   −−−▷             Forward(2, 3, False, False, <StorageType.FWD_RESTART: 2>)
                  5  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  6  .   .   .   ◀−−−         Reverse(4, 3, True)
                  7                           Copy(2, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  8  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                  9  .   .   ◀−−−             Reverse(3, 2, True)
                 10                           Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                 11  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                 12  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 13  .   ◀−−−                 Reverse(2, 1, True)
                 14                           Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                 15  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 16  ◀−−−                     Reverse(1, 0, True)
                 17  End Reverse              EndReverse(False,)


Now, let us modify the storage type to ``'RAM'`` of the additional
forward restart checkpointing by setting the optional
:class:`TwoLevelCheckpointSchedule` argument
``binomial_storage = StorageType.RAM``. Thus, on the example above, ones
notices that the action associated with ``Action index: 8`` implies the
forward restart data storage should be on ``'disk'``. On the other hand,
the example below displays that the action associated to
``Action index: 8`` indicates that the forward restart data storage
should be in ``'RAM'``.

.. code:: ipython3

    revolver = TwoLevelCheckpointSchedule(3, binomial_snapshots=snaps_on_disk, 
                                          binomial_storage=StorageType.RAM)
    solver_manager.execute(revolver)


.. parsed-literal::

      Action index:  Run-time illustration    Action:
    ---------------  -----------------------  ------------------------------------------------------------
                  0  +−−−▷−−−▷−−−▷            Forward(0, 3, True, False, <StorageType.DISK: 1>)
                  1  .   .   .   +−−−▷        Forward(3, 6, True, False, <StorageType.DISK: 1>)
                  2  End Forward              EndForward()
                  3                           Copy(3, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  4  .   .   .   −−−▶         Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
                  5  .   .   .   ◀−−−         Reverse(4, 3, True)
                  6                           Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                  7  −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
                  8  .   +−−−▷                Forward(1, 2, True, False, <StorageType.RAM: 0>)
                  9  .   .   −−−▶             Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
                 10  .   .   ◀−−−             Reverse(3, 2, True)
                 11                           Move(1, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
                 12  .   −−−▶                 Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
                 13  .   ◀−−−                 Reverse(2, 1, True)
                 14                           Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
                 15  −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
                 16  ◀−−−                     Reverse(1, 0, True)
                 17  End Reverse              EndReverse(False,)


References
----------

[1] Griewank, A., & Walther, A. (2000). Algorithm 799: revolve: an
implementation of checkpointing for the reverse or adjoint mode of
computational differentiation. ACM Transactions on Mathematical Software
(TOMS), 26(1), 19-45., doi: https://doi.org/10.1145/347837.347846

[2] Stumm, P., & Walther, A. (2009). Multistage approaches for optimal
offline checkpointing. SIAM Journal on Scientific Computing, 31(3),
1946-1967. https://doi.org/10.1137/080718036

[3] Aupy, G., Herrmann, J., Hovland, P., & Robert, Y. (2016). Optimal
multistage algorithm for adjoint computation. SIAM Journal on Scientific
Computing, 38(3), C232-C255. DOI: https://doi.org/10.1145/347837.347846.

[4] Aupy, G., & Herrmann, J. (2017). Periodicity in optimal hierarchical
checkpointing schemes for adjoint computations. Optimization Methods and
Software, 32(3), 594-624. doi:
https://doi.org/10.1080/10556788.2016.1230612

[5] Herrmann, J. and Pallez (Aupy), G. (2020). H-Revolve: a framework
for adjoint computation on synchronous hierarchical platforms. ACM
Transactions on Mathematical Software (TOMS), 46(2), 1-25. DOI:
https://doi.org/10.1145/3378672.

[6] Maddison, J. R. (2023). On the implementation of checkpointing with
high-level algorithmic differentiation. arXiv preprint arXiv:2305.09568.
https://doi.org/10.48550/arXiv.2305.09568.

[7] Pringle, G. C., Jones, D. C., Goswami, S., Narayanan, S. H. K., and
Goldberg, D. (2016). Providing the ARCHER community with adjoint
modelling tools for high-performance oceanographic and cryospheric
computation. https://nora.nerc.ac.uk/id/eprint/516314.

[8] Goldberg, D. N., Smith, T. A., Narayanan, S. H., Heimbach, P., and
Morlighem, M. (2020). Bathymetric Influences on Antarctic Ice‐Shelf Melt
Rates. Journal of Geophysical Research: Oceans, 125(11), e2020JC016370.
doi: https://doi.org/10.1029/2020JC016370.
