from checkpoint_schedules import HRevolveCheckpointSchedule
    # (
    # # MemoryCheckpointSchedule,
    # #  PeriodicDiskCheckpointSchedule,
    # #  MultistageCheckpointSchedule,
    # #  TwoLevelCheckpointSchedule,
     
    # #  MixedCheckpointSchedule
    # )
from checkpoint_schedules import \
     (Write, Clear, Configure,
     Forward, EndForward, Reverse, 
     Read, EndReverse, WriteForward)
import functools
import time as tm
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
    def __init__(self, forward, backward, total_steps, save_ram=None,
                 save_disk=None, period=None, schedule='hrevolve'):
        self.save_ram = save_ram
        self.save_disk = save_disk
        self.forward = forward
        self.backward = backward
        self.tot_steps = total_steps
        self.schedule = schedule
        self.period = period
        self.action_list = []

    def cp_schedule(self):
        """Return the schedule.
        """
        if self.schedule == 'hrevolve':
            return HRevolveCheckpointSchedule(
                self.tot_steps, snapshots_in_ram=self.save_ram,
                snapshots_on_disk=self.save_disk)
        # elif self.schedule == 'disk_revolve':
        #     assert self.save_disk is not None
        #     return RevolveCheckpointSchedule(
        #         self.tot_steps, snapshots_on_disk=self.save_disk,
        #         revolve_sequence=self.schedule)
        # elif self.schedule == 'periodic_disk_revolve':
        #     assert self.save_ram is not None
        #     return RevolveCheckpointSchedule(
        #         self.tot_steps, snapshots_in_ram=self.save_ram,
        #         revolve_sequence=self.schedule)
        # elif self.schedule == 'periodic_disk':
        #     assert self.period is not None
        #     return PeriodicDiskCheckpointSchedule(self.period)
        # elif self.schedule == 'two_level':
        #     assert self.period is not None
        #     return TwoLevelCheckpointSchedule(self.period, self.save_disk)
        # elif self.schedule == 'mixed':
        #     return MixedCheckpointSchedule(self.tot_steps, self.save_disk)
        # elif self.schedule == 'multistage':
        #     return MultistageCheckpointSchedule(self.tot_steps, self.save_ram, self.save_disk)

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

        @action.register(Clear)
        def action_clear(cp_action):
            if cp_action.clear_ics:
                ics.clear()
            if cp_action.clear_data:
                data.clear()

        @action.register(Configure)
        def action_configure(cp_action):
            nonlocal store_ics, store_data

            store_ics = cp_action.store_ics
            store_data = cp_action.store_data

        @action.register(Write)
        def action_write(cp_action):
            snapshots[cp_action.storage][cp_action.n] = (set(ics), set(data))

        @action.register(WriteForward)
        def action_write_forward(cp_action):
            print("---")
            # assert len(ics) == 0 and len(data) > 0
            # assert cp_action.n == max(data)
            
        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n

            self.forward.advance(cp_action.n0, cp_action.n1)

            n1 = min(cp_action.n1, self.tot_steps)
            model_n = n1
            if store_ics:
                ics.update(range(cp_action.n0, n1))
            if store_data:
                data.update(range(cp_action.n0, n1+1))
            if n1 == self.tot_steps:
                cp_schedule.finalize(n1)

        @action.register(Reverse)
        def action_reverse(cp_action):
            nonlocal model_r
            self.backward.advance(cp_action.n1, cp_action.n0)
            model_r += cp_action.n1 - cp_action.n0

        @action.register(Read)
        def action_read(cp_action):
            nonlocal model_n
            cp = snapshots[cp_action.storage][cp_action.n]
            model_n = None

            if len(cp[0]) > 0:
                ics.clear()
                ics.update(cp[0])
                model_n = cp_action.n

            if len(cp[1]) > 0:
                data.clear()
                data.update(cp[1])
            
            if cp_action.delete:
                del snapshots[cp_action.storage][cp_action.n]

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
        store_ics = False
        ics = set()
        store_data = False
        data = set()

        snapshots = {"RAM": {}, "disk": {}}
        cp_schedule = self.cp_schedule()
        if cp_schedule is None:
            print("Incompatible with schedule type")

        assert cp_schedule.n() == 0
        assert cp_schedule.r() == 0
        assert (
                cp_schedule.max_n() is None
                or cp_schedule.max_n() == self.tot_steps
            )
        while True:
            cp_action = next(cp_schedule)
            self.action_list.append(cp_action)
            action(cp_action)
            assert model_n is None or model_n == cp_schedule.n()
            assert model_r == cp_schedule.r()

            if isinstance(cp_action, EndReverse):
                # print(self.action_list, len(self.action_list))
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
        print((">"*(n_1-n_0)).rjust(n_1))
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
     PeriodicDiskCheckpointSchedule,
        n1
            Initial time step in reverse state.
        n0
            Final time step in reverse state.

        """
        print("<".rjust(n_1))
        i_n = n_1
        while i_n > n_0:
            i_np1 = i_n - 1
            i_n = i_np1

schedule_list = ['hrevolve', 'periodic_disk_revolve', 'disk_revolve', 'periodic_disk',
                 'multistage', 'two_level', 'mixed']

start = tm.time()
steps = 4
schk = 2
sdisk = 0
fwd = execute_fwd()
bwd = execute_bwd()
manage = Manage(fwd, bwd, steps, save_ram=schk, save_disk=sdisk, schedule='hrevolve')
manage.actions()
# manage = Manage(fwd, bwd, steps, period=2, schedule='periodic_disk')
# manage.actions()
# manage = Manage(fwd, bwd, steps, save_ram=3, schedule='periodic_disk_revolve')
# manage.actions()
# manage = Manage(fwd, bwd, steps, save_disk=4, period=2, schedule='disk_revolve')
# manage.actions()
# manage = Manage(fwd, bwd, steps, save_disk=4, schedule='mixed')
# manage.actions()
# # manage = Manage(fwd, bwd, steps, save_ram=3, save_disk=1, schedule='multistage')
# manage.actions()
# manage.actions()
# end = tm.time()

print(end-start)


