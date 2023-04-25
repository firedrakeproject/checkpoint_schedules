#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from checkpoint_schedules import \
    (HRevolveCheckpointSchedule, Write, Clear, Configure,
     Forward, EndForward, Reverse, Read, EndReverse)
import functools

__all__ = \
    [
        "Manage"
    ]


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
    save_chk : int
        Number of checkpoint that will be stored.
    total_steps : int
        Total steps used to execute the solvers.

    """
    def __init__(self, forward, backward, save_chk, total_steps):
        self.save_chk = save_chk
        self.forward = forward
        self.backward = backward
        self.tot_steps = total_steps
        self.action_list = []

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

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal model_n

            self.forward.advance(cp_action.n0, cp_action.n1)

            n1 = min(cp_action.n1, self.tot_steps)
            model_n = n1
            if store_ics:
                ics.update(range(cp_action.n0, n1))
            if store_data:
                data.update(range(cp_action.n0, n1))
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
            # assert model_r == self.tot_steps

            if not cp_action.exhausted:
                model_r = 0

        S = (self.save_chk,)
        for s in S:
            model_n = 0
            model_r = 0

            store_ics = False
            ics = set()
            store_data = False
            data = set()

            snapshots = {"RAM": {}, "disk": {}}

            hrev_schedule = HRevolveCheckpointSchedule(self.tot_steps, self.save_chk, 0)

            if hrev_schedule is None:
                print("Incompatible with schedule type")

            assert hrev_schedule.n() == 0
            assert hrev_schedule.r() == 0
            assert (
                    hrev_schedule.max_n() is None
                    or hrev_schedule.max_n() == self.tot_steps
                   )
            count = 0
            while True:
                cp_action = next(hrev_schedule)
                self.action_list.append(cp_action)
                action(cp_action)
                count += 1
                print(count)
                assert model_n is None or model_n == hrev_schedule.n()
                assert model_r == hrev_schedule.r()

                if isinstance(cp_action, EndReverse):
                    break
