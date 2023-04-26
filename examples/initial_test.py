#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import time as tm
from checkpoint_schedules import HRevolveCheckpointSchedule

class FowardSolver():
    """Define the a forward solver.

    """
    def __init__(self):
        self.exp = None
        self.chk_id = None
        self.steps = steps
        self.chk = None

    def advance(self, schedules):
        """Advance the foward equation.

        Parameters
        ----------
        n0
            Initial time step.
        n1
            Final time step.

        """
        print("Start forward.")
        i_n = 0
        i_w = 0
        index_w = []
        for sch in schedules:
            if sch.type == "Write":
                index_w.append(sch.index[1])
        while i_n < self.steps:
            if len(index_w) != 0 and i_n == index_w[0]:
                if index_w[0] != 0:
                    print((">"*(i_n-i_w)).rjust(i_n))
                print("Writing checkpoint at step: ", i_n)
                i_w = index_w[0]
                index_w.pop(0)
            i_np1 = i_n + 1
            
            i_n = i_np1
        print((">"*(i_n-i_w)).rjust(i_n))
        print("End forward.")

class BackwardSolver():
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
steps = 10
schk = 3
hrev_schedule = HRevolveCheckpointSchedule(steps, schk, 0)

print("hrevolve schedule: ", hrev_schedule._schedule)
fwd_schedule = hrev_schedule.get_forward_schedule()
bwd_schedule = hrev_schedule.get_reverse_schedule()
print("Forward hrevolve schedule: ", fwd_schedule)
print("Forward hrevolve schedule: ", bwd_schedule)
# quit()
# fwd = FowardSolver()
# fwd.advance(fwd_schedule)
