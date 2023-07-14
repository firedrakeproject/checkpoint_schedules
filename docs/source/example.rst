
.. _example_checkpoint_schedules:
Example
-------

The *checkpoint_schedules* package offers a squedule of actions that
enable the coordination of an adjoint-based gradient executions through
a checkpoint strategy. The checkpoint schedule is built by the sequence
of actions referred to as :class:`~checkpoint_schedules.schedule.Forward`, 
:class:`~checkpoint_schedules.schedule.EndForward`, 
:class:`~checkpoint_schedules.schedule.Reverse`, 
:class:`~checkpoint_schedules.schedule.Copy`,
:class:`~checkpoint_schedules.schedule.EndReverse`. The actions provides functionalities 
such as storing the forward checkpoint data used to restart the forward solver, 
storing the forward checkpoint data for adjoint computations, and retrieving the
stored data for both the forward solver restart and the adjoint
computation.

In the following code, we have implemented the ``CheckpointingManager``
class, which allows the manegement of the forward and adjoint executions
in time. On using ``CheckpointingManager.execute`` method, we iterate
over a sequence of actions given by the schedule ``cp_schedule``. The
actions are defined by using single-dispatch functions, where the
``action`` function is the generic function using the singledispatch
decorator. Specific functions for different types of
*checkpoint_schedules* actions are provided by using the register method
of the base function ``action``.

.. code:: ipython3

    from checkpoint_schedules import Forward, EndForward, Reverse, Copy, EndReverse
    import functools
    
    class CheckpointingManager():
        """Manage the forward and adjoint solvers.
    
        Attributes
        ----------
        adj_grad_problem : object
            Adjoint-based gradient object.
        save_ram : int
            Number of checkpoint that will be stored in RAM.
        save_disk : int
            Number of checkpoint that will be stored on disk.
        list_actions : list
            Store the list of actions.
        max_n : int
            Total steps used to execute the solvers.
        """
        def __init__(self, max_n, save_ram, save_disk):
            self.max_n = max_n
            self.save_ram = save_ram
            self.save_disk = save_disk
            self.list_actions = []
            
        def execute(self, cp_schedule):
            """Execute forward and adjoint with a checkpointing strategy.
    
            Parameters
            ----------
            cp_schedule : object
                Checkpointing schedule.
            """
            @functools.singledispatch
            def action(cp_action):
                raise TypeError("Unexpected action")
    
            @action.register(Forward)
            def action_forward(cp_action):
                nonlocal model_n
                if cp_action.write_ics:
                    print(("*").rjust(cp_action.n0*4))
    
                print(("|" + "--->"*(cp_action.n1-cp_action.n0)).rjust(cp_action.n1*4) +
                       "   "*(self.max_n - cp_action.n0) + 
                       self.list_actions[len(self.list_actions) - 1])
    
                n1 = min(cp_action.n1, self.max_n)
                model_n = n1
                if cp_action.n1 == self.max_n:
                    cp_schedule.finalize(n1)
    
            @action.register(Reverse)
            def action_reverse(cp_action):
                nonlocal model_r
                print(("<---"*(cp_action.n1-cp_action.n0) + "|").rjust(cp_action.n1*4) 
                      + "   "*(self.max_n - cp_action.n0) + 
                        self.list_actions[len(self.list_actions) - 1])
    
                model_r += cp_action.n1 - cp_action.n0
                
            @action.register(Copy)
            def action_copy(cp_action):
                print(("+").rjust(cp_action.n*4) 
                      + " "*(self.max_n - cp_action.n) + 
                        self.list_actions[len(self.list_actions) - 1])
    
        
            @action.register(EndForward)
            def action_end_forward(cp_action):
                assert model_n == self.max_n
                print("End Forward")
                
            @action.register(EndReverse)
            def action_end_reverse(cp_action):
                nonlocal model_r
                assert model_r == self.max_n
                print("End Reverse")
    
            model_n = 0
            model_r = 0
    
            count = 0
            while True:
                cp_action = next(cp_schedule)
                self.list_actions.append(str(cp_action))
                action(cp_action)
                count += 1
                if isinstance(cp_action, EndReverse):  
                    break

Firstly, let us define the total steps used in the computations, and the
number of steps that the checkpoint data is going to be saved in RAM and
disk.

.. code:: ipython3

    max_n = 4 # Total number of time steps.
    save_ram = 1 # Number of steps to save i RAM.
    save_disk = 1 # Number of steps to save in disk.

Next, let us set the ``CheckpointingManager`` manage object with the
attibutes defined above.

.. code:: ipython3

    chk_manager = CheckpointingManager(max_n, save_ram, save_disk) # manager object

The *checkpoint_schedules* package is already working with multistage
[2], the two-level mixed periodic/binomial [6], and H-Revolve [4]
schedules that were originally implemented by [4]. However, we carried
out minor modifications to reach sequence of operation attending our
approach, i. e., the schedule has explicity actions of writing and
deleting the data required for the forward solver restarting and for the
use in the adjoint computation. If you want more details of the revolver
algorithimics, fell free to access the API documentation avaiable here.

Additionally, *checkpoint_schedules* provides an iterator that convert
revolver operations into the *checkpoint_schedules* format.

Before to obtain the schedule, we need to define the revolver approach.
Below we choose the schedule iterator ``revolver`` with the H-Revolve
approach.

.. code:: ipython3

    from checkpoint_schedules import HRevolve
    s = 0
    revolver = HRevolve(max_n, save_ram, save_disk)

We then define the sequence by ``revolver.sequence()`` and execute the
forward and adjoint solver with ``chk_manager.execute(revolver)``.

.. code:: ipython3

    revolver.sequence()
    chk_manager.execute(revolver)


.. parsed-literal::

    *
    |--->--->--->            Forward(0, 3, True, False, 'RAM')
                |--->        Forward(3, 4, False, True, 'RAM')
    End Forward
                <---|        Reverse(4, 3, True)
    +                        Copy(0, 'RAM', False)
    |--->--->                Forward(0, 2, False, False, 'NONE')
            |--->            Forward(2, 3, False, True, 'RAM')
            <---|            Reverse(3, 2, True)
    +                        Copy(0, 'RAM', False)
    |--->                    Forward(0, 1, False, False, 'NONE')
        |--->                Forward(1, 2, False, True, 'RAM')
        <---|                Reverse(2, 1, True)
    +                        Copy(0, 'RAM', True)
    |--->                    Forward(0, 1, False, True, 'RAM')
    <---|                    Reverse(1, 0, True)
    End Reverse


The output above illustrates how it works the forward and adjoint
executions in time with the *checkpoint_schedules* package. The symbol
``|`` indicates the step that the solver initialises. The symbom ``*``
on top of ``|`` indicates that the data used to restart the forward
solver is stored. Whereas the symbol ``+`` indicates the action of
copying the storage data and be used as initial condition for the
forward solver recomputations.

To complement the illustration above, consider some of the actions
explained as follow:

-  Forward action

   -  General form: *``Forward``\ (n0, n1, write_ics, write_adj_deps,
      ‘storage’)*
   -  Particular form:

      -  *Forward(0, 3, True, False, ‘RAM’)*:

         -  Execute the forward solver from step 0 to step 3.
         -  Write the forward data (*write_ics* is True) of step 0 to
            RAM (storage).
         -  The forward data is not stored for the adjoint computation
            (*write_adj_deps* is False).

      -  *Forward(3, 4, False, True, ‘RAM’)*:

         -  Execute the forward solver from step 3 to step 4.
         -  Do not write the forward data (*write_ics* is False) of step
            4.
         -  Store the forward data for the adjoint computation
            (*write_adj_deps* is *True*) in RAM (storage).

-  Reverse action

   -  General form: *Reverse(n1, n0, clear_adj_deps)*
   -  Particular form:

      -  *Reverse(4, 3, True)*:

         -  Execute the adjoint solver from step 4 to step 3.
         -  Clear the adjoint dependencies (*clear_adj_deps* is True)
            used in the adjoint computation.

-  Copy action

   -  General form: Copy(n, from_storage, delete)
   -  Particular form:

      -  Copy(2, ‘RAM’, True):

         -  Copy the forward data related to step 2 from RAM.
         -  Delete the copied data from RAM (*delete* is *True*) as it
            is not needed anymore to restart the forward solver.


Below we have the schedule obtained from Disk-Revolve and
Periodic-Disk-Revolve checkpoint strategies.

.. code:: ipython3

    from checkpoint_schedules import DiskRevolve
    revolver = DiskRevolve(max_n, save_ram, save_disk)
    print(revolver._schedule)
    revolver.sequence()
    chk_manager.execute(revolver)


.. parsed-literal::

    None
    *
    |--->--->--->            Forward(0, 3, True, False, 'RAM')
                |--->        Forward(3, 4, False, True, 'RAM')
    End Forward
                <---|        Reverse(4, 3, True)
    +                        Copy(0, 'RAM', False)
    |--->--->                Forward(0, 2, False, False, 'NONE')
            |--->            Forward(2, 3, False, True, 'RAM')
            <---|            Reverse(3, 2, True)
    +                        Copy(0, 'RAM', False)
    |--->                    Forward(0, 1, False, False, 'NONE')
        |--->                Forward(1, 2, False, True, 'RAM')
        <---|                Reverse(2, 1, True)
    +                        Copy(0, 'RAM', True)
    |--->                    Forward(0, 1, False, True, 'RAM')
    <---|                    Reverse(1, 0, True)
    End Reverse


.. code:: ipython3

    from checkpoint_schedules import PeriodicDiskRevolve
    revolver = PeriodicDiskRevolve(max_n, save_ram, save_disk)
    revolver.sequence(period=2)
    chk_manager.execute(revolver)


.. parsed-literal::

    We use periods of size  2
    *
    |--->--->            Forward(0, 2, True, False, 'DISK')
           *
           |--->         Forward(2, 3, True, False, 'RAM')
               |--->     Forward(3, 4, False, True, 'RAM')
    End Forward
               <---|     Reverse(4, 3, True)
           +             Copy(2, 'RAM', True)
           |--->         Forward(2, 3, False, True, 'RAM')
           <---|         Reverse(3, 2, True)
    +                    Copy(0, 'DISK', False)
    *
    |--->                Forward(0, 1, True, False, 'RAM')
       |--->             Forward(1, 2, False, True, 'RAM')
       <---|             Reverse(2, 1, True)
    +                    Copy(0, 'RAM', True)
    |--->                Forward(0, 1, False, True, 'RAM')
    <---|                Reverse(1, 0, True)
    End Reverse


This first example gives the basics of executions involving an
adjoint-based gradient using *checkpoint_schedules* package. The next
:ref:`section <tutorial_checkpoint_schedules>` shows an example an application 
of adjoint-based gradient problem.
