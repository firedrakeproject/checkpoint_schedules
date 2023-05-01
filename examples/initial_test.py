#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time as tm
from checkpoint_schedules import HRevolveCheckpointSchedule


def solvers_with_hrevolve(fwd, bwd, chk_in_ram, chk_in_disk, steps):
    """Manage the forward and backward solvers.
    This object manage the solvers with the employment of the
    hrevolve checkpointing.

    Parameters
    ----------
    fwd : object
        The forward solver.
    bwd : object
        The backward solver.
    chk_in_ram : int
        Number of checkpoint stored in RAM.
    chk_in_disk : int
        Number of checkpoint stored in Disk.
    steps : int
        Total steps used to execute the solvers.
    """
    S = (chk_in_ram,)
    for s in S:
        model_n = 0
        model_r = 0
        hrev_schedule = HRevolveCheckpointSchedule(steps, chk_in_ram, chk_in_disk)

        store_ics = False
        store_data = False
        ics = set()
        data = set()
        snapshots = {"RAM": {}, "disk": {}}

        if hrev_schedule is None:
            print("Incompatible with schedule type")

        assert hrev_schedule.n() == 0
        assert hrev_schedule.r() == 0
        assert (
                hrev_schedule.max_n() is None
                or hrev_schedule.max_n() == steps
            )
        while True:
            cp_action = next(hrev_schedule)
            if cp_action.type == "Clear":
                if cp_action.clear_ics:
                    ics.clear()
                if cp_action.clear_data:
                    data.clear()
            elif cp_action.type == "Configure":
                store_ics = cp_action.store_ics
                store_data = cp_action.store_data
            elif cp_action.type == "Write":
                assert ics is not None
                snapshots[cp_action.storage][cp_action.n] = (set(ics), set(data))
            elif cp_action.type == "Forward":
                fwd.advance(cp_action.n0, cp_action.n1)
                n1 = min(cp_action.n1, steps)
                model_n = n1
                if store_ics:
                    ics.add(n1)
                if store_data:
                    data.add(n1)
                if n1 == steps:
                    hrev_schedule.finalize(n1)
            elif cp_action.type == "Read":
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
            elif cp_action.type == "Reverse":
                bwd.advance(cp_action.n1, cp_action.n0)
                model_r += cp_action.n1 - cp_action.n0
            elif cp_action.type == "EndForward":
                assert model_n is not None and model_n == steps
            elif cp_action.type == "EndReverse":
                assert model_r == steps
                if not cp_action.exhausted:
                    model_r = 0

            # action_list.append(cp_action)           
            assert model_n is None or model_n == hrev_schedule.n()
            assert model_r == hrev_schedule.r()
            if cp_action.type == "EndReverse":
                break


class Forward():
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
 
           
    def getsteps(self) -> int:
        """Return the total time steps.

        """
        return steps
   

class Backward():
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
steps = 5
schk = 2
fwd = Forward()
bwd = Backward()
solvers_with_hrevolve(fwd, bwd, schk, 0, steps)
end = tm.time()
print(end-start)