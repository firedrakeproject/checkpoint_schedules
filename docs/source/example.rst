.. _example_checkpoint_schedules:
Example
-------

To exemplify the *checkpoint_schedules* usage in adjoint-based gradient
problems, let us consider a class that illustrates the implementation of
an adjoint-based gradient. Below is the ``GradAdj`` class, which
includes the forward and backward methods to execute respectively the
forward and adjoint systems. The ``copy_fwd_data`` function carries
forward data copying from either RAM or disk. This data is then used as
the initial condition in the forward solver restarting. The functions
``store_ram`` and ``store_disk`` are responsible for storing the forward
data for restarting purposes. Additionally, the ``store_adj_deps``
function is responsible for storing the forward data required for the
adjoint computation.

.. code:: ipython3

    class GradAdj():
        """This class illustrates an adjoint-based gradient computation.
        """
        def __init__(self):
            self.snapshots = {'RAM': {}, 'DISK': {}}
            self.adj_deps = {}
            self.adj_tape = {}
            self.fwd_tape = {0: 0} #illustrating an initial condition at step 0
        
      
        def forward(self, n0, n1, write_ics=False, write_adj_deps=False, storage=None):
            """Execute a forward solver in time.
    
            Parameters
            ----------
            n0 : int
                Initial step.
            n1 : int
                Final step.
            write_ics : bool, optional
                Write the forward data to disk or RAM. 
                This data will be used as the initial condition for the forward solver restarting.
            write_adj_deps : bool, optional
                Write the forward data to disk or RAM. This data is an adjoint dependence.
            """
            data_n = self.fwd_tape[n0]
            if write_ics:
                if storage == 'RAM':
                    self.store_in_ram(data_n, n0)
                elif storage == 'DISK':
                    self.store_on_disk(data_n, n0)
            
            steps = int(n1 - n0)
            t = 0
            while t < steps:
                data_np1 = data_n + 1
                data_n = data_np1
                t += 1
                
            self.fwd_tape.clear()    
            self.fwd_tape = {n1: data_np1}
            
            if write_adj_deps:
                self.store_adj_deps(data_np1, n1)
    
    
        def backward(self, n0, n1, clear_adj_deps):
            """Execute the adjoint solver in time.
    
            Parameters
            ----------
            n0 : int
                Initial time step.
            n1 : int
                Final time step.
            clear_adj_deps : bool
                Clear the adjoint dependencies.
            """
            bwd_data_n = self.adj_tape[n1]
            steps = int(n1 - n0)
            t = 0
            while t < steps:
                bwd_data_np1 = bwd_data_n - 1
                bwd_data_n = bwd_data_np1
                t += 1
            self.adj_tape = {n0: bwd_data_n}
            if clear_adj_deps:
                self.adj_deps.clear()
            
    
        def copy_fwd_data(self, n, from_storage, delete):
            """Copy the forward data from RAM or disk to the forward tape.
            
            Parameters
            ----------
            n : int
                Time step.
            from_storage : str
                Storage type.
            delete : bool
                Delete the forward data stored either in RAM or disk.
            """
            if from_storage == 'DISK':
                u0 = self.snapshots[from_storage][n]
            else:
                u0 = self.snapshots[from_storage][n]
            
            self.fwd_tape.clear()
            self.fwd_tape = {n: u0}
            
            if delete:
                del self.snapshots[from_storage][n]
        
    
        def store_in_ram(self, data, step):
            """Store the forward data in RAM.
    
            Parameters
            ----------
            data : array
                Forward data.
            step : int
                Time step.
            """
            self.snapshots['RAM'][step] = data
    
    
        def store_on_disk(self, data, step):
            """Store the forward data on disk.
    
            Parameters
            ----------
            data : array
                Forward data.
            step : int
                Time step.
            """
            self.snapshots['DISK'][step] = data
            
        def store_adj_deps(self, data, n):
            """Store the adjoint dependencies.
            
            Parameters
            ----------
            data : array
                Adjoint dependencies.
            n : int
                Time step.
            """
            self.adj_deps = {n: data}
        
        def adj_initcondition(self, data, n):
            self.adj_tape = {n: data}
      

Using *checkpoint_schedules* package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *checkpoint_schedules* package offers a squedule of actions that
enable the coordination of an adjoint-based gradient executions through
a checkpoint strategy. The checkpoint schedule is built by the sequence
of actions referred to as :class:`~checkpoint_schedules.schedule.Forward`, 
:class:`~checkpoint_schedules.schedule.EndForward`, 
:class:`~checkpoint_schedules.schedule.Reverse`, 
:class:`~checkpoint_schedules.schedule.Copy`,
:class:`~checkpoint_schedules.schedule.EndReverse`. The actions provides functionalities such as storing the
forward checkpoint data used to restart the forward solver, storing the
forward checkpoint data for adjoint computations, and retrieving the
stored data for both the forward solver restart and the adjoint
computation. Additionally, *checkpoint_schedules* provides an iterator
that convert revolver operations into the *checkpoint_schedules* format.

In the following code, we have implemented the ``CheckpointingManager``
class, which allows the manegement of the forward and adjoint executions
in time. With ``CheckpointingManager.execute`` method, we iterate over a
sequence of actions given by the schedule ``cp_schedule``. The actions
are defined by using single-dispatch functions, where the ``action``
function is the generic function using the singledispatch decorator.
Specific functions for different types of *checkpoint_schedules* actions
are provided by using the register method of the base function
``action``.

.. code:: ipython3

    from checkpoint_schedules import Forward, EndForward, Reverse, Copy, EndReverse, StorageLocation
    import functools
    from colorama import Fore, Back, Style
    
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
        def __init__(self, adj_grad_problem, max_n, save_ram, save_disk):
            self.max_n = max_n
            self.save_ram = save_ram
            self.save_disk = save_disk
            self.adj_grad_problem = adj_grad_problem
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
                print(("|" + "--->"*(cp_action.n1-cp_action.n0)).rjust(cp_action.n1*4))
                self.adj_grad_problem.forward(cp_action.n0, cp_action.n1, 
                                              write_ics=cp_action.write_ics, 
                                              write_adj_deps=cp_action.write_adj_deps,
                                              storage=cp_action.storage)
    
                n1 = min(cp_action.n1, self.max_n)
                model_n = n1
                if cp_action.n1 == self.max_n:
                    cp_schedule.finalize(n1)
    
            @action.register(Reverse)
            def action_reverse(cp_action):
                nonlocal model_r
                print(("<---"*(cp_action.n1-cp_action.n0) + "|").rjust(cp_action.n1*4))
                self.adj_grad_problem.backward(cp_action.n0, cp_action.n1, 
                                               cp_action.clear_adj_deps)
                model_r += cp_action.n1 - cp_action.n0
                
            @action.register(Copy)
            def action_copy(cp_action):
                self.adj_grad_problem.copy_fwd_data(cp_action.n, 
                                                    cp_action.from_storage, 
                                                    cp_action.delete)
        
            @action.register(EndForward)
            def action_end_forward(cp_action):
                assert model_n == self.max_n
                self.adj_grad_problem.adj_initcondition(model_n, model_n)
                
            @action.register(EndReverse)
            def action_end_reverse(cp_action):
                nonlocal model_r
                assert model_r == self.max_n
    
            model_n = 0
            model_r = 0
    
            storage_limits = {StorageLocation(0).name: self.save_ram, 
                              StorageLocation(1).name: self.save_disk}
    
            count = 0
            print("|---"*(max_n) + "|")
            while True:
                print()
                cp_action = next(cp_schedule)
                action(cp_action)
                self.list_actions.append([count, str(cp_action)])
                count += 1
                if isinstance(cp_action, EndReverse):  
                    break

Firstly, let us define the adjoint-based gradient object. Also, set the
total steps used in the computations, the number of steps that the
checkpoint data is going to be saved in RAM and disk.

.. code:: ipython3

    grad_adj = GradAdj() # Defining the adjoint-based gradient.
    max_n = 4 # Total number of time steps.
    save_ram = 1 # Number of steps to save i RAM.
    save_disk = 1 # Number of steps to save in disk.

Next, let us set the ``CheckpointingManager`` manage object, where one
of the attributes is the adjoint-based gradient object, ``grad_adj``.

.. code:: ipython3

    chk_manager = CheckpointingManager(grad_adj, max_n, save_ram, save_disk) # manager object

The *checkpoint_schedules* package has already provided revolver
algorithimics originally implemented by . However, using the API
required to solver adjoint-based gradient problem that makes explicit
the storing, copying and deleting of the data required for the forward
solver restarting and for the use in the adjoint computation. For
technical details involving the revolver algorithimics, please access
the documentation.

Below we choose to build the checkpoint schedule trhough on H-Revolve
algorithmic,

.. code:: ipython3

    from checkpoint_schedules import HRevolve
    revolver = HRevolve(max_n, save_ram, snap_on_disk=save_disk)

We then set obtain the sequence by ``revolver.sequence()`` and execute
the forward and adjoint solver with checkpointing methods with
``chk_manager.execute(revolver)``.

.. code:: ipython3

    revolver.sequence()
    chk_manager.execute(revolver)


.. parsed-literal::

    |---|---|---|---|
    
    [34m|--->--->
    
    [34m       |--->
    
    [34m           |--->
    
    
    [31m           <---|
    
    
    [34m       |--->
    
    [31m       <---|
    
    
    [34m|--->
    
    [34m   |--->
    
    [31m   <---|
    
    
    [34m|--->
    
    [31m<---|
    


The output above illustrates how it works the forward and adjoint
executions with the *checkpoint_schedules* package. The symbol ``|``
indicates the step that the solver initialise. Hence,

from tabulate import tabulate print(Fore.BLACK +
tabulate(chk_manager.list_actions, headers=[“Action number”,
“checkpoint_schedules actions”]))

Below we presented the role of the *checkpoint_schedules* actions for
some of the cases:

-  Action number 0: *Forward(0, 2, True, False, ‘RAM’)*:

   -  Execute the forward solver from step 0 to step 2.
   -  Write the forward data (*write_ics* is True) of step 0 to RAM
      (storage).
   -  The forward data is not stored for the adjoint computation
      (*write_adj_deps* is False).

-  Action number 2: *Forward(3, 4, False, True, ‘RAM’)*:

   -  Execute the forward solver from step 3 to step 4.
   -  Do not write the forward data (*write_ics* is False) of step 4.
   -  Store the forward data for the adjoint computation
      (*write_adj_deps* is *True*) in RAM (storage).

-  Action number 4: *Reverse(4, 3, True)*:

   -  Execute the adjoint solver from step 4 to step 3.
   -  Clear the adjoint dependencies (*clear_adj_deps* is True) used in
      the adjoint computation.

-  Action number 5: Copy(2, ‘RAM’, ‘TAPE’, True):

   -  Copy the forward data related to step 2 from RAM to TAPE.
   -  Delete the copied data from RAM (*delete* is *True*) as it is not
      needed anymore to restart the forward solver.

-  Action number 8: Copy(0, ‘DISK’, ‘TAPE’, True):

   -  Copy the forward data related to step 0 from DISK to TAPE.
   -  Do not delete the copied data from DISK (*delete* is *FALSE*).


:ref:`tutorial <tutorial_checkpoint_schedules>`