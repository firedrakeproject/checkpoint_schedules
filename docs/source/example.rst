.. _example_checkpoint_schedules:

Using *checkpoint_schedules*
----------------------------

The examples aims to introduce the usage of checkpoint_schedules through
an initial illustration of how this package works and how to interpret
the run time of forward and adjoint solvers using a schedule of the
*checkpointing_schedules* package.

*checkpointing_schedules* is designed to be a versatile package that can
hold both schedules using checkpointing methods and cases where no
checkpointing strategies are employed. In the latter case, this package
can provide a schedule when no adjoint computation is required, which
leads no need forward checkpointing data storage. Furthermore, for the
case where all time-steps forward checkpoint data used for the adjoint
computation are stored either in memory or in disk.

The schedule is given by a sequence of *checkpoint_schedules* actions:
*Forward*, *Reverse*, *EndForward*, *EndReverse*, *Move*, and *Copy*. \*
*Forward*: indicate the forward solver execution. Additionally, it can
manage the forward checkpoint data storage. \* *Reverse*: indicate the
adjoint solver execution. \* *EndForward* and *EndReverse*: Indicate the
end of the forward and adjoint solvers, respectively. \* *Move*:
Indicate movement forward checkpoint data storage from one storage type
to another. \* *Copy*: Indicate the copy of forward checkpoint data from
one storage type to another.

In the next sections of this example, we will explain how to employ
these actions in the forward and adjoint solvers.

Manager of the forward and adjoint executions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us consider the ``CheckpointingManager`` class, which plays the
essential role in managing forward and adjoint executions. This
management is given by iterating over a schedule with the execution of
``CheckpointingManager.execute(cp_schedule)``. To perform this task, the
method requires a ``cp_schedule`` argument, which must be a
*checkpoint_schedules object* containing a schedule attibute and a
generator method. This generator is responsible for yielding the
*checkpoint_schedules* actions to be executed.

Whitin ``CheckpointingManager.execute``, we implement the
*checkpointing_schedules* actions using single-dispatch functions. The
base function, ``action``, is decorated with the ``singledispatch``.
Specific actions functions are then established by using the register
method of the ``action``. Hence, ``CheckpointingManager.execute``
effectively calls the specific action according to a schedule.

.. code:: ipython3

    from checkpoint_schedules import Forward, EndForward, Reverse, Copy, Move, EndReverse, StorageType
    import functools
    
    class CheckpointingManager():
        """Manage the executions of the forward and adjoint solvers.
    
        Attributes
        ----------
        max_n : int
            Total steps used to execute the solvers.
        chk_ram : int, optional
            Number of checkpoint stored on `'RAM'`.
        chk_disk : int, optional
            Number of checkpoint stored on `'DISK'`.
        """
        def __init__(self, max_n, chk_ram=0, chk_disk=0):
            self.max_n = max_n
            self.save_ram = chk_ram
            self.save_disk = chk_disk
            self.list_actions = []
            
        def execute(self, cp_schedule):
            """Execute forward and adjoint with a checkpointing schedule.
    
            Parameters
            ----------
            cp_schedule : CheckpointSchedule
                Checkpointing schedule object.
            """
            @functools.singledispatch
            def action(cp_action):
                raise TypeError("Unexpected action")
    
            @action.register(Forward)
            def action_forward(cp_action):
                def illustrate_runtime(a, b, singlestorage):
                    if singlestorage:
                        print(((a + '\u2212\u2212' + b)*(n1-cp_action.n0)).rjust(n1*4) +
                       "   "*(self.max_n - n1 + 4) + 
                       self.list_actions[len(self.list_actions) - 1])
                    else:
                        print((a + ('\u2212\u2212\u2212' + b)*(n1-cp_action.n0)).rjust(n1*4) +
                        "   "*(self.max_n - n1 + 4) + 
                        self.list_actions[len(self.list_actions) - 1])
    
                nonlocal model_n
                n1 = min(cp_action.n1, self.max_n)
                if cp_action.write_ics and cp_action.write_adj_deps:
                    a = '\u002b'
                    b = "\u25b6"
                    singlestorage = True
                else:
                    singlestorage = False
                    if cp_action.write_ics:
                        a = '\u002b'
                    else:
                        a = ''
                    if cp_action.write_adj_deps:
                        b = "\u25b6"
                    else:
                        b = "\u25b7"
                illustrate_runtime(a, b, singlestorage)
                
                model_n = n1
                if n1 == self.max_n:
                    # Imposing the latest time step.
                    # It is required for the online schedule.
                    cp_schedule.finalize(n1)
    
            @action.register(Reverse)
            def action_reverse(cp_action):
                nonlocal model_r
                print((('\u25c0' + '\u2212\u2212\u2212')*(cp_action.n1-cp_action.n0)).rjust(cp_action.n1*4) 
                      + "   "*(self.max_n - cp_action.n1 + 4) + 
                        self.list_actions[len(self.list_actions) - 1])
                model_r += cp_action.n1 - cp_action.n0
                
            @action.register(Copy)
            def action_copy(cp_action):
                print("    "*(self.max_n + 4) + 
                        self.list_actions[len(self.list_actions) - 1])
    
            @action.register(Move)
            def action_move(cp_action):
                print("    "*(self.max_n + 4) + 
                        self.list_actions[len(self.list_actions) - 1])
    
            @action.register(EndForward)
            def action_end_forward(cp_action):
                assert model_n == self.max_n
                # The correct number of adjoint steps has been taken
                print("End Forward" + "   "*(self.max_n + 2) + 
                        self.list_actions[len(self.list_actions) - 1])
                if cp_schedule._max_n is None:
                    cp_schedule._max_n = self.max_n
                
            @action.register(EndReverse)
            def action_end_reverse(cp_action):
                nonlocal model_r
                assert model_r == self.max_n
                print("End Reverse" + "   "*(self.max_n + 3) + 
                      self.list_actions[len(self.list_actions) - 1])
    
            model_n = 0
            model_r = 0
    
            for count, cp_action in enumerate(cp_schedule):
                self.list_actions.append(str(cp_action))
                action(cp_action)
                if isinstance(cp_action, EndReverse):  
                    break

Schedule without adjoint computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When adjoint computation is unnecessary, only ``max_n`` (representing
the maximum number of steps) is the essential argument. To reach an
comprehensive example, we define the ``CheckpointingManager`` object
with a short number of steps, ``max_n=4``.

.. code:: ipython3

    max_n = 4 # Total number of time steps.
    solver_manager = CheckpointingManager(max_n) # manager object

In this current case, we present the ``NoneCheckpointSchedule`` object,
which efficiently manage the forward solver execution without the need
for adjoint computation and without any data storage.

.. code:: ipython3

    from checkpoint_schedules import NoneCheckpointSchedule
    cp_schedule = NoneCheckpointSchedule() # Create a object where no adjoint computation is needed.
    solver_manager.execute(cp_schedule) # Execute the solver according the schedule.


.. parsed-literal::

    −−−▷−−−▷−−−▷−−−▷            Forward(0, 9223372036854775807, False, False, <StorageType.NONE: None>)
    End Forward                  EndForward()


The output of the ``solver_manager.execute(cp_schedule)`` function
illustrates the execution over time on the right side, while displaying
*checkpoint_schedules* actions on the left side.

The general *Forward* action explaining can be read in the ``Forward``
class. In the following, we will explain the *Forward* action in the
context of the *NoneCheckpointSchedule*.

-  *Forward(0, 9223372036854775807, False, False, <StorageType.NONE:
   None>)*:

   -  Advance the forward solver from step ``n0 = 0`` to the start of
      any step ``n1``.
   -  Both ``write_ics`` and ``write_adj_deps`` are set to ``'False'``,
      indicating that the forward solver does not store the forward
      restart data and the forward data required for the adjoint
      computation.
   -  The storage type is ``StorageType.NONE``, indicating that no
      specific storage type is required.
   -  The ``EndForward()`` action indicates that the forward solver has
      reached the end of the time interval.

   This type of schedule is referred to as online since it does not
   require specifying a value for the maximum steps to obtain the
   schedule. Hence, users can define any desired step as needed. For
   this specific case, we set the maximum step with ``max_n = 4``, which
   is an attribute of the CheckpointingManager. When then inform the
   schedule the final step by executing the following code below into
   the ``action_forward`` function.
   ``python   n1 = min(cp_action.n1, self.max_n)   ...   cp_schedule.finalize(n1)``
   The ``cp_schedule.finalize(n1)`` function can be set according the
   user’s requirement and is used to inform the schedule that the
   forward solver has reached the end of the time interval.

Schedule for all time-steps forward data storage (no checkpointing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code is valuable for the cases where the user intend to
store the forward checkpoint data for all time-steps and does not
requires any checkpointing strategy provided by the ‘revolvers’ methods.
This schedule is given by using the ``SingleStorageSchedule`` class. The
code below shows how to used this class in the forward and adjoint
computations.

.. code:: ipython3

    from checkpoint_schedules import SingleMemoryStorageSchedule
    
    cp_schedule = SingleMemoryStorageSchedule()
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    −−−▶−−−▶−−−▶−−−▶            Forward(0, 9223372036854775807, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
    ◀−−−◀−−−◀−−−◀−−−            Reverse(4, 0, True)
    End Reverse                     EndReverse(False,)


In this particular case, we have:

-  *Forward(0, 9223372036854775807, True, True, <StorageType.RAM: 0>)*,
   which reads:

   -  Advance the forward solver from the step ``n0`` to the start of
      any step ``n1``.
   -  Do not store the forward restart data once if ``write_ics`` is
      ``'True'``.
   -  Store the forward data required for the adjoint computation once
      ``write_adj_deps`` is ``'True'``.
   -  Storage type is ``<StorageType.ADJ_DEPS: 3>``, which can be read
      as a “variable” that holds the adjoint dependency data (forward
      data) used for the adjoint computation.

-  *Reverse(4, 0, True)*
-  Advance the forward solver from the step ``4`` to the start of the
   step ``0``.
-  Clear the adjoint dependency data once ``clear_adj_deps`` is
   ``'True'``.
-  *EndReverse(True)* indicates that the reverse actions reached the end
   of the time interval.

This schedule does not focus on storing the forward restart data since
there is no need to restart the forward solver. However, if the user
requires storing the forward restart data for all time steps, the
``SingleMemoryStorageSchedule`` class can sort it with the attributes
``write_ics`` and ``storage_ics``. By setting ``write_ics`` to ``True``
and ``storage_ics`` to ``True``, as illustrated in the following
example, the user can effectively store the forward restart data for all
time steps.

.. code:: ipython3

    from checkpoint_schedules import StorageType
    print("Store the forward restart data for all steps in Disk")
    cp_schedule = SingleMemoryStorageSchedule(write_ics=True, storage_ics=StorageType.DISK)
    solver_manager.execute(cp_schedule)
    
    print(" ")
    print("Store the forward restart data for all steps in RAM")
    cp_schedule = SingleMemoryStorageSchedule(write_ics=True, storage_ics=StorageType.RAM)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    Store the forward restart data for all steps in Disk
    +−−▶+−−▶+−−▶+−−▶            Forward(0, 9223372036854775807, True, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
    ◀−−−◀−−−◀−−−◀−−−            Reverse(4, 0, True)
    End Reverse                     EndReverse(False,)
     
    Store the forward restart data for all steps in RAM
    +−−▶+−−▶+−−▶+−−▶            Forward(0, 9223372036854775807, True, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
    ◀−−−◀−−−◀−−−◀−−−            Reverse(4, 0, True)
    End Reverse                     EndReverse(False,)


Schedules for Checkpointing Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let us consider the checkpoint schedules given by a checkpointing
strategy. We will begin by exploring the Revolve approach, as introduced
in reference [1].

The Revolve checkpointing strategy involves storing checkpoints only in
``'RAM'``. So, we now set the attribute ``chk_ram`` of the
``CheckpointingManager`` class to specify the number of steps at which
the forward restart data should be stored in ``'RAM'``.

The schedule is generated by the ``Revolve`` class, which takes the
number of steps at which the adjoint data should be stored in ``'RAM'``
as an argument. This number of steps is specified by the attribute
``chk_ram``.

The code below shows how to generate a schedule for the ``Revolve``
checkpointing strategy and illustrates the execution of the forward and
adjoint computations using the generated Revolve schedule.

.. code:: ipython3

    from checkpoint_schedules import Revolve
    chk_ram = 2 
    solver_manager = CheckpointingManager(max_n, chk_ram=chk_ram) # manager object
    cp_schedule = Revolve(max_n, chk_ram) 
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    +−−−▷−−−▷                  Forward(0, 2, True, False, <StorageType.RAM: 0>)
           +−−−▷               Forward(2, 3, True, False, <StorageType.RAM: 0>)
                −−−▶            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
                ◀−−−            Reverse(4, 3, True)
                                    Move(2, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
            −−−▶               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
            ◀−−−               Reverse(3, 2, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
        −−−▶                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
        ◀−−−                  Reverse(2, 1, True)
                                    Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
    ◀−−−                     Reverse(1, 0, True)
    End Reverse                     EndReverse()


In the following code illustrated the forward and adjoint computations
using the checkpointing given by HRevolve strategy [2]. In this case,
the storage of the forward data is done in\ ``'RAM'`` and on ``'disk'``.
The checkpointing schedule is generated with ``HRevolve`` class.

The ``HRevolve`` class requires the following parameters: maximum steps
stored in RAM (``snap_in_ram``), maximum steps stored on disk
(``snap_on_disk``), and the number of time steps (``max_n``). Thus, we
firs import the necessary module ``HRevolve`` from the
*checkpoint_schedules* package. Then, ``HRevolve`` class is instantiated
with the following parameters: ``snap_in_ram = 1``,
``snap_on_disk = 1``, and ``max_n = 4``. Finally, we illustrate the
forward and adjoint executions the H-Revolve checkpointing strategy
based.

.. code:: ipython3

    from checkpoint_schedules import HRevolve
    chk_disk = 1
    chk_ram = 1
    solver_manager = CheckpointingManager(max_n, chk_ram=chk_ram, chk_disk=chk_disk) # manager object
    cp_schedule = HRevolve(max_n, chk_ram, snap_on_disk=chk_disk)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    +−−−▷−−−▷−−−▷               Forward(0, 3, True, False, <StorageType.RAM: 0>)
                −−−▶            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
                ◀−−−            Reverse(4, 3, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷−−−▷                  Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
            −−−▶               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
            ◀−−−               Reverse(3, 2, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
        −−−▶                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
        ◀−−−                  Reverse(2, 1, True)
                                    Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
    ◀−−−                     Reverse(1, 0, True)
    End Reverse                     EndReverse()


Let us consider some of the actions that are printed above:

-  *Forward(0, 3, True, False, <StorageType.RAM: 0>)* reads:

   -  Advance the forward solver from time step 0 to the start of the
      step 3.
   -  Store the forward data required to restart the forward solver at
      the start of step 0.
   -  Do not store the forward data required for the adjoint computation
      once ``write_adj_deps`` is ``'False'``.
   -  Store the forward restart data in ``'RAM'``.

-  *Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)* reads:

   -  Advance the forward solver from time step 3 to the start of the
      step 4.
   -  Do not store the forward data required to restart the forward
      solver at the start of step 3 (``write_ics`` is False).
   -  Store the forward data required for the adjoint computation
      (``write_adj_deps`` is ``'True'``).
   -  Store the forward for the adjoint computation in a ‘variable’ that
      holds the adjoint dependency required for the adjoint computation.

-  *Restart(4, 3, True)* reads:

   -  Restart the forward solver from the start of step 4 to the start
      of step 3.
   -  Clear the the adjoint dependency variable (``clear_adj_deps`` is
      ``'True'``).

-  *Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)* reads:

   -  Copy the forward data stored in ``'RAM'`` to the ‘variable’ that
      holds the data required to restart the forward solver at the start
      of step 0.

-  *Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)* reads:

   -  Move the forward data stored in ``'RAM'`` to the ‘variable’ that
      holds the data required to restart the forward solver at the start
      of step 0. The data in ``'RAM'`` is cleared after the move once
      the data is no longer required.

The code below illustrates the forward and adjoint computations for
other checkpointing strategies available in the package. The output are
similar to the one above.

The complete list of checkpointing strategies available in the package
are: \* No checkpointing: without adjoint computation. No forward data
is stored. \* Single checkpointing storage: the forward data required
for the adjoint computation is stored in all the time steps. \* Revolve
checkpointing: the storage is only stored in ``'RAM'``. For additional
details, see `Revolve
checkpointing <https://doi.org/10.1016/j.jcp.2018.12.039>`__. \*
Periodic disk checkpointing: the forward data required for the adjoint
computation is stored in a periodic fashion. For additional details, see
`Periodic checkpointing <https://doi.org/10.1016/j.jcp.2018.12.039>`__.
\* Disk checkpointing: the forward data required for the adjoint
computation is stored in ``'RAM'`` and in ``'DISK'``. For additional
details, see `Disk
checkpointing <https://doi.org/10.1016/j.jcp.2018.12.039>`__. \*
Multi-level checkpointing: the forward data required for the adjoint
computation is stored in ``'RAM'`` and in ``'DISK'``. For additional
details, see `Multi-level
checkpointing <https://doi.org/10.1016/j.jcp.2018.12.039>`__. \*
Two-level checkpointing: the forward data required for the adjoint
computation is stored in ``'RAM'`` and in ``'DISK'``. For additional
details, see `Two-level
checkpointing <https://doi.org/10.1016/j.jcp.2018.12.039>`__. \* Mixed
checkpointing: the forward data required for the adjoint computation is
stored in ``'RAM'`` and in ``'DISK'``. For additional details, see
`Mixed checkpointing <https://doi.org/10.1016/j.jcp.2018.12.039>`__.

.. code:: ipython3

    from checkpoint_schedules import DiskRevolve
    solver_manager = CheckpointingManager(max_n, chk_ram=chk_ram) # manager object``
    cp_schedule = DiskRevolve(max_n, chk_ram)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    `'+−−−▷−−−▷−−−▷               Forward(0, 3, True, False, <StorageType.RAM: 0>)
                −−−▶            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
                ◀−−−            Reverse(4, 3, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷−−−▷                  Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
            −−−▶               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
            ◀−−−               Reverse(3, 2, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
        −−−▶                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
        ◀−−−                  Reverse(2, 1, True)
                                    Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
    ◀−−−                     Reverse(1, 0, True)
    End Reverse                     EndReverse()'`


.. code:: ipython3

    from checkpoint_schedules import PeriodicDiskRevolve
    cp_schedule = PeriodicDiskRevolve(max_n, chk_ram)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    `'We use periods of size  3
    +−−−▷−−−▷−−−▷               Forward(0, 3, True, False, <StorageType.RAM: 0>)
                −−−▶            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
                ◀−−−            Reverse(4, 3, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷−−−▷                  Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
            −−−▶               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
            ◀−−−               Reverse(3, 2, True)
                                    Copy(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
        −−−▶                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
        ◀−−−                  Reverse(2, 1, True)
                                    Move(0, <StorageType.RAM: 0>, <StorageType.FWD_RESTART: 2>)
    −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
    ◀−−−                     Reverse(1, 0, True)
    End Reverse                     EndReverse()'`


.. code:: ipython3

    from checkpoint_schedules import MixedCheckpointSchedule
    
    cp_schedule = MixedCheckpointSchedule(max_n, chk_disk)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    `'+−−−▷−−−▷−−−▷               Forward(0, 3, True, False, <StorageType.DISK: 1>)
                −−−▶            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
                ◀−−−            Reverse(4, 3, True)
                                    Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    −−−▷−−−▷                  Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
            −−−▶               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
            ◀−−−               Reverse(3, 2, True)
                                    Move(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    −−−▶                     Forward(0, 1, False, True, <StorageType.DISK: 1>)
        −−−▶                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
        ◀−−−                  Reverse(2, 1, True)
                                    Move(0, <StorageType.DISK: 1>, <StorageType.ADJ_DEPS: 3>)
    ◀−−−                     Reverse(1, 0, True)
    End Reverse                     EndReverse()'`


.. code:: ipython3

    from checkpoint_schedules import MultistageCheckpointSchedule
    
    cp_schedule = MultistageCheckpointSchedule(max_n, 0, chk_disk)
    solver_manager.execute(cp_schedule)


.. parsed-literal::

    +−−−▷−−−▷−−−▷               Forward(0, 3, True, False, <StorageType.DISK: 1>)
                −−−▶            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
    End Forward                  EndForward()
                ◀−−−            Reverse(4, 3, True)
                                    Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    −−−▷−−−▷                  Forward(0, 2, False, False, <StorageType.FWD_RESTART: 2>)
            −−−▶               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
            ◀−−−               Reverse(3, 2, True)
                                    Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    −−−▷                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
        −−−▶                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
        ◀−−−                  Reverse(2, 1, True)
                                    Move(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    −−−▶                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
    ◀−−−                     Reverse(1, 0, True)
    End Reverse                     EndReverse()


.. code:: ipython3

    from checkpoint_schedules import TwoLevelCheckpointSchedule
    revolver = TwoLevelCheckpointSchedule(2, binomial_snapshots=2)
    solver_manager.execute(revolver)


.. parsed-literal::

    `'|--->--->                  Forward(0, 2, True, False, <StorageType.DISK: 1>)
           |--->--->            Forward(2, 4, True, False, <StorageType.DISK: 1>)
    End Forward                  EndForward()
           c                        Copy(2, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
           |--->               Forward(2, 3, False, False, <StorageType.FWD_RESTART: 2>)
               |--->            Forward(3, 4, False, True, <StorageType.ADJ_DEPS: 3>)
               <---|            Reverse(4, 3, True)
           c                        Copy(2, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
           |--->               Forward(2, 3, False, True, <StorageType.ADJ_DEPS: 3>)
           <---|               Reverse(3, 2, True)
    c                        Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    |--->                     Forward(0, 1, False, False, <StorageType.FWD_RESTART: 2>)
       |--->                  Forward(1, 2, False, True, <StorageType.ADJ_DEPS: 3>)
       <---|                  Reverse(2, 1, True)
    c                        Copy(0, <StorageType.DISK: 1>, <StorageType.FWD_RESTART: 2>)
    |--->                     Forward(0, 1, False, True, <StorageType.ADJ_DEPS: 3>)
    <---|                     Reverse(1, 0, True)
    End Reverse                  EndReverse(False,)'`


References
~~~~~~~~~~

[1] Andreas Griewank and Andrea Walther, ‘Algorithm 799: revolve: an
implementation of checkpointing for the reverse or adjoint mode of
computational differentiation’, ACM Transactions on Mathematical
Software, 26(1), pp. 19–45, 2000, doi: 10.1145/347837.347846

[2] Herrmann, J. and Pallez (Aupy), G.. “H-Revolve: a framework for
adjoint computation on synchronous hierarchical platforms.” ACM
Transactions on Mathematical Software (TOMS) 46.2 (2020): 1-25. doi:
10.1145/3378672

[3] Aupy, G., Herrmann, Ju. and Hovland, P. and Robert, Y. “Optimal
multistage algorithm for adjoint computation”. SIAM Journal on
Scientific Computing, 38(3), C232-C255, (2016). DOI:
https://doi.org/10.1145/347837.347846

