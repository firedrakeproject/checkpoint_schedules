from checkpoint_schedules import RevolveCheckpointSchedule, StorageLocation
    # (
    # MemoryCheckpointSchedule,
    #  PeriodicDiskCheckpointSchedule,
    #  MultistageCheckpointSchedule,
    #  TwoLevelCheckpointSchedule,
     
    #  MixedCheckpointSchedule
    # )
from checkpoint_schedules import \
     (Forward, EndForward, Reverse, Transfer, EndReverse)
import functools
# import time as tm
from tabulate import tabulate
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Manage():
    """Manage the forward and backward solvers.

    This object manage the solvers with the employment of the
    hrevolve checkpointing.

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
    def __init__(self, forward, backward, total_steps, schedule='hrevolve', save_ram=0,
                 save_disk=0, period=None):
        self.save_ram = save_ram
        self.save_disk = save_disk
        self.forward = forward
        self.backward = backward
        self.tot_steps = total_steps
        self.period = period
        self.action_list = []
        self.c = 0
        self.c_back = total_steps
        self.schedule = schedule

    def actions(self):
        """Actions.

        Raises
        ------
        TypeError
            _description_
        """
        @functools.singledispatch
        def action(cp_action):
            raise TypeError("Unexpected action")

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n
            self.c += (cp_action.n1 - cp_action.n0)
            self.forward.advance(cp_action.n0, cp_action.n1)
            # self.action_list.append([self.c, cp_action])
            n1 = min(cp_action.n1, self.tot_steps)
            model_n = n1
            
            ics.clear()
            if cp_action.write_ics:
                ics.update(range(cp_action.n0, n1))
                snapshots[cp_action.storage][cp_action.n0] = set(ics)
            if cp_action.write_data:
                data.update(range(cp_action.n0, n1))
            
            if n1 == self.tot_steps:
                cp_schedule.finalize(n1)

        @action.register(Reverse)
        def action_reverse(cp_action):
            # self.action_list.append([self.c_back, cp_action])
            self.c_back -= 1
            nonlocal model_r
            self.backward.advance(cp_action.n1, cp_action.n0)
            model_r += cp_action.n1 - cp_action.n0
            if cp_action.clear_fwd_data:
                data.clear()

        @action.register(Transfer)
        def action_transfer(cp_action):
            # pass
            nonlocal model_n
            model_n = None
            if cp_action.delete:
                del snapshots[cp_action.from_storage][cp_action.n]
            elif cp_action.to_storage != StorageLocation(2).name:
                assert cp_action.n in snapshots[cp_action.from_storage]
                assert cp_action.n < self.tot_steps - model_r
                # No data is currently stored for this step
                assert cp_action.n not in ics
                assert cp_action.n not in data
                snapshots[cp_action.to_storage][cp_action.n] = snapshots[cp_action.from_storage][cp_action.n]
            else:
                data.update(range(cp_action.n, cp_action.n + 1))

        @action.register(EndForward)
        def action_end_forward(cp_action):
            # The correct number of forward steps has been taken
            assert model_n is not None and model_n == self.tot_steps

        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            nonlocal model_r

            # The correct number of adjoint steps has been taken
            assert model_r == self.tot_steps

            if not cp_action.exhausted:
                model_r = 0

        model_n = 0
        model_r = 0
        ics = set()
        data = set()

        snapshots = {StorageLocation(0).name: {}, StorageLocation(1).name: {}}
        cp_schedule = RevolveCheckpointSchedule(self.tot_steps, self.save_ram,
                                                snap_on_disk=self.save_disk)
        print(cp_schedule._schedule)
        
        snapshots = {StorageLocation(0).name: {}, StorageLocation(1).name: {}}
        
        if cp_schedule is None:
            print("Incompatible with schedule type")

        assert cp_schedule.n() == 0
        assert cp_schedule.r() == 0
        assert (
                cp_schedule.max_n() is None
                or cp_schedule.max_n() == self.tot_steps
            )
        
        c = 0
        while True:
            cp_action = next(cp_schedule)
            self.action_list.append([c, cp_action.info()])
            action(cp_action)
            assert model_n is None or model_n == cp_schedule.n()
            assert model_r == cp_schedule.r()
            c += 1
            if isinstance(cp_action, EndReverse):
                # col_names = ["Index", "Actions"]
                # #display table
                # print(snapshots)
                print(tabulate(self.action_list))  
                break


class execute_fwd():
    """Define the a forward solver.

    """
    def __init__(self):
        self.exp = None
        self.chk_id = None
        self.steps = steps
        self.chk = None

    def advance(self, n_0: int, n_1: int) -> None:
        """Advance the foward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        # print((">"*(n_1-n_0)).rjust(n_1))
        i_n = n_0
        while i_n < n_1:
            i_np1 = i_n + 1
            i_n = i_np1
        self.chk = i_n
           
    def getsteps(self) -> int:
        """Return the total time steps.

        """
        return self.steps
   

class execute_bwd():
    """This object define the a forward solver.

    """
    def __init__(self):
        self.exp = None
        self.sol = None

    def advance(self, n_1: int, n_0: int) -> None:
        """Execute the backward equation.

        Parameters
        ----------
        n1
            Initial time step in reverse state.
        n0
            Final time step in reverse state.

        """
        # print("<".rjust(n_1))
        i_n = n_1
        while i_n > n_0:
            i_np1 = i_n - 1
            i_n = i_np1


# start = tm.time()
steps = 7
schk = 2
sdisk = 1
fwd = execute_fwd()
bwd = execute_bwd()
manage = Manage(fwd, bwd, steps, save_ram=schk, save_disk=sdisk)
manage.actions()




