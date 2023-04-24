#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import time as tm
from checkpoint_schedules import \
    (HRevolveCheckpointSchedule, Write,
     Forward, EndForward, Reverse, Read, EndReverse)


class Manage():
    """Manage the backward solver with checkpoint schedules.

    This object manage the solvers with the employment of the
    hrevolve checkpointing.

    Attributes
    ----------
    forward : object
        The forward solver.
    backward : object
        The backward solver.
    hrevolve_chedules : object
        Hrevolve schdules of the reverse mode.
    """
    def __init__(self, forward, backward, hrevolve_schedules):
        self.forward = forward
        self.backward = backward
        self.hrev_schedule = hrevolve_schedules
        self.action_list = []

    def actions(self):
        """Actions of checkpoint scheduled.
        """
        @functools.singledispatch
        def action(cp_action):
            raise TypeError("Unexpected action")

        @action.register(Write)
        def action_write(cp_action):
            print("Write the checkpoint.")

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n

            self.forward.advance(cp_action.n0, cp_action.n1)
            n1 = min(cp_action.n1, self.hrev_schedule.max_n)
            model_n = n1

        @action.register(Reverse)
        def action_reverse(cp_action):
            nonlocal model_r
            self.backward.advance(cp_action.n1, cp_action.n0)
            model_r += cp_action.n1 - cp_action.n0

        @action.register(Read)
        def action_read(cp_action):
            nonlocal model_n
            model_n = None
            print("Read the checkpoint at step: ", cp_action.n)

        @action.register(EndForward)
        def action_end_forward(cp_action):
            """Action end forward.
            """
            assert model_n is not None and model_n == self.hrev_schedule.max_n

        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            """Action end reverse.
            """
            nonlocal model_r

            # The correct number of adjoint steps has been taken
            assert model_r == self.hrev_schedule.max_n

            if not cp_action.exhausted:
                model_r = 0

        model_n = hrev_schedule._max_n
        model_r = 0

        if hrev_schedule is None:
            print("Incompatible with schedule type")

        assert hrev_schedule.n() == hrev_schedule._max_n
        assert hrev_schedule.r() == 0
        while True:
            cp_action = next(hrev_schedule)
            self.action_list.append(cp_action)
            assert model_n is None or model_n == hrev_schedule.n()
            assert model_r == hrev_schedule.r()

            if isinstance(cp_action, EndReverse):
                break


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
fwd_schedule = hrev_schedule.get_forward_schedule()
bwd_schedule = hrev_schedule.get_reverse_schedule()

fwd = FowardSolver()
fwd.advance(fwd_schedule)


# bwd_schedule = 
# cp_action = next(hrev_schedule)
# self.action_list.append(cp_action)
# action(cp_action)
# fwd = Forward()
bwd = BackwardSolver()
manage = Manage(fwd, bwd, hrev_schedule)

manage.actions()
# end = tm.time()
# print(end-start)