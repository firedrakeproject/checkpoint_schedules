from checkpoint_schedules import \
    (HRevolveCheckpointSchedule, Write, Clear, Configure,
     Forward, EndForward, Reverse, Read, EndReverse, WriteForward)
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
    def __init__(self, forward, backward, save_ram, save_disk, total_steps):
        self.save_ram = save_ram
        self.save_disk = save_disk
        self.forward = forward
        self.backward = backward
        self.tot_steps = total_steps

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
            assert len(ics) == 0 and len(data) > 0
            assert cp_action.n == max(data)
            
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
                hrev_schedule.finalize(n1)

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

        hrev_schedule = HRevolveCheckpointSchedule(self.tot_steps, self.save_ram, self.save_disk)
        if hrev_schedule is None:
            print("Incompatible with schedule type")

        assert hrev_schedule.n() == 0
        assert hrev_schedule.r() == 0
        assert (
                hrev_schedule.max_n() is None
                or hrev_schedule.max_n() == self.tot_steps
            )
        while True:
            cp_action = next(hrev_schedule)
            action(cp_action)
            assert model_n is None or model_n == hrev_schedule.n()
            assert model_r == hrev_schedule.r()

            if isinstance(cp_action, EndReverse):
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


start = tm.time()
steps = 200
schk = 7
sdisk = 7
fwd = execute_fwd()
bwd = execute_bwd()
manage = Manage(fwd, bwd, schk, sdisk, steps)
manage.actions()
end = tm.time()
print(end-start)