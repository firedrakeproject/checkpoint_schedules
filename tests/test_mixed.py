#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import pytest

from checkpoint_schedules import (
    MixedCheckpointSchedule, Copy, Forward, Reverse, EndForward, EndReverse,
    Move, StorageType)
from checkpoint_schedules.mixed import (
    optimal_steps_mixed, mixed_step_memoization)


@pytest.mark.parametrize("n, S", [
                                  (1, (0,)),
                                  (2, (1,)),
                                  (3, (1, 2)),
                                  (5, (2,)),
                                  (10, tuple(range(2, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))
                                  ])
def test_mixed(n, S):
    @functools.singledispatch
    def action(cp_action):
        raise TypeError("Unexpected action")

    @action.register(Forward)
    def action_forward(cp_action):
        nonlocal model_n, model_steps

        ics.clear()
        data.clear()

        # Start at the current location of the forward
        assert model_n is not None and cp_action.n0 == model_n
        # Do not advance further than the current location of the adjoint
        assert cp_action.n1 <= n - model_r

        if cp_action.write_ics:
            # Advance at least two steps when storing forward restart data
            assert cp_action.n1 > cp_action.n0 + 1
            # Do not advance further than one step before the current location
            # of the adjoint
            assert cp_action.n1 < n - model_r

            assert cp_action.storage == StorageType.DISK
            assert cp_schedule.uses_storage_type(StorageType.DISK)
            ics.update(range(cp_action.n0, cp_action.n1))
            assert cp_action.n0 not in snapshots
            snapshots[cp_action.n0] = (set(ics), set(data))

        if cp_action.write_adj_deps:
            # Advance exactly one step when storing non-linear dependency data
            assert cp_action.n1 == cp_action.n0 + 1
            # Do not advance further than the current location of the adjoint
            assert cp_action.n1 <= n - model_r

            data.update(range(cp_action.n0, cp_action.n1))
            if cp_action.storage == StorageType.DISK:
                assert cp_schedule.uses_storage_type(StorageType.DISK)
                assert cp_action.n0 not in snapshots
                snapshots[cp_action.n0] = (set(ics), set(data))
            else:
                assert cp_action.storage == StorageType.WORK

        # Stored data consists of either forward restart or non-linear
        # dependency data, but not both
        assert not cp_action.write_ics or not cp_action.write_adj_deps

        model_n = cp_action.n1
        model_steps += cp_action.n1 - cp_action.n0

    @action.register(Reverse)
    def action_reverse(cp_action):
        nonlocal model_r

        # Start at the current location of the adjoint
        assert cp_action.n1 == n - model_r
        # Advance exactly one step
        assert cp_action.n0 == cp_action.n1 - 1
        # Non-linear dependency data for the step is stored
        assert cp_action.n0 in data

        if cp_action.clear_adj_deps:
            data.clear()

        model_r += 1

    @action.register(Copy)
    def action_copy(cp_action):
        nonlocal model_n

        # The checkpoint exists
        assert cp_action.n in snapshots
        assert cp_action.from_storage == StorageType.DISK

        cp = snapshots[cp_action.n]

        # No data is currently stored
        assert len(ics) == 0
        assert len(data) == 0
        # The checkpoint contains forward restart data
        assert len(cp[0]) > 0 and len(cp[1]) == 0

        # Loading a forward restart checkpoint:

        # The checkpoint data is at least two steps away from the current
        # location of the adjoint
        assert cp_action.n < n - model_r - 1
        # The loaded data is deleted iff non-linear dependency data for all
        # remaining steps can be stored
        assert cp_action.n < n - model_r - 1 - (s - len(snapshots) + 1)

        assert cp_action.to_storage == StorageType.WORK
        ics.update(cp[0])

        model_n = cp_action.n

        # Can advance the forward to the current location of the adjoint
        assert ics.issuperset(range(model_n, n - model_r))

    @action.register(Move)
    def action_move(cp_action):
        nonlocal model_n

        # The checkpoint exists
        assert cp_action.n in snapshots
        assert cp_action.from_storage == StorageType.DISK

        cp = snapshots.pop(cp_action.n)

        # No data is currently stored
        assert len(ics) == 0
        assert len(data) == 0
        # The checkpoint contains either forward restart or non-linear
        # dependency data, but not both
        assert len(cp[0]) == 0 or len(cp[1]) == 0
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        if len(cp[0]) > 0:
            # Loading a forward restart checkpoint:

            # The checkpoint data is at least two steps away from the current
            # location of the adjoint
            assert cp_action.n < n - model_r - 1
            # The loaded data is deleted iff non-linear dependency data for all
            # remaining steps can be stored
            assert cp_action.n >= n - model_r - 1 - (s - len(snapshots) + 1)

            assert cp_action.to_storage == StorageType.WORK
            ics.update(cp[0])

            model_n = cp_action.n

            # Can advance the forward to the current location of the adjoint
            assert ics.issuperset(range(model_n, n - model_r))

        if len(cp[1]) > 0:
            # Loading a non-linear dependency data checkpoint:

            # The checkpoint data is exactly one step away from the current
            # location of the adjoint
            assert cp_action.n == n - model_r - 1

            assert cp_action.to_storage == StorageType.WORK
            data.update(cp[1])

            model_n = None

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        # The correct number of adjoint steps has been taken
        assert model_r == n
        # The schedule has concluded
        assert cp_schedule.is_exhausted

    for s in S:
        print(f"{n=:d} {s=:d}")

        model_n = 0
        model_r = 0
        model_steps = 0

        ics = set()
        data = set()

        snapshots = {}

        cp_schedule = MixedCheckpointSchedule(n, s, storage=StorageType.DISK)
        assert n == 1 or cp_schedule.uses_storage_type(StorageType.DISK)
        assert cp_schedule.n == 0
        assert cp_schedule.r == 0
        assert cp_schedule.max_n == n

        for _, cp_action in enumerate(cp_schedule):
            action(cp_action)
            # The schedule state is consistent with both the forward and
            # adjoint
            assert model_n is None or model_n == cp_schedule.n
            assert model_r == cp_schedule.r
            assert cp_schedule.max_n == n

            # Either no data is stored, or exactly one of forward restart or
            # non-linear dependency data is stored
            assert len(ics) == 0 or len(data) == 0
            # Non-linear dependency data is stored for at most one step
            assert len(data) <= 1
            # Checkpoint storage limits are not exceeded
            assert len(snapshots) <= s

        # The correct total number of forward steps has been taken
        assert model_steps == optimal_steps_mixed(n, s)
        assert model_steps == mixed_step_memoization(n, s)[2]
        # No data is stored
        assert len(ics) == 0 and len(data) == 0
        # No checkpoints are stored
        assert len(snapshots) == 0

        # The schedule has concluded
        assert cp_schedule.is_exhausted
        try:
            next(cp_schedule)
        except StopIteration:
            pass
        except Exception:
            raise RuntimeError("Iterator not exhausted")
