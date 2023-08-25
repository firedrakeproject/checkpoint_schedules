#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import pytest

from checkpoint_schedules import MixedCheckpointSchedule, Copy, \
     Forward, Reverse, EndForward, EndReverse, Move, StorageType
from checkpoint_schedules.mixed import optimal_steps_mixed, \
    mixed_step_memoization


@pytest.mark.parametrize("n, S", [
                                  (1, (0,)),
                                  (2, (1,)),
                                  (3, (1, 2)),
                                  (10, tuple(range(1, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))
                                  ])
def test_mixed(n, S):
    cp_action_lists = []

    @functools.singledispatch
    def action(cp_action):
        raise TypeError("Unexpected action")

    @action.register(Forward)
    def action_forward(cp_action):
        nonlocal model_n, model_steps
        nonlocal store_ics, store_data
        store_ics = cp_action.write_ics
        store_data = cp_action.write_adj_deps
        ics.clear()
        data.clear()

        # Start at the current location of the forward
        assert model_n is not None and model_n == cp_action.n0

        if store_ics:
            assert cp_action.storage == StorageType.DISK
            assert cp_schedule.uses_storage_type(StorageType.DISK)
            # Advance at least two steps when storing forward restart data
            assert cp_action.n1 > cp_action.n0 + 1
            # Do not advance further than one step before the current location
            # of the adjoint
            assert cp_action.n1 < n - model_r
            # No data for these steps is stored
            assert len(ics.intersection(range(cp_action.n0, cp_action.n1))) == 0  # noqa: E501

        if store_data:
            # Advance exactly one step when storing non-linear dependency data
            assert cp_action.n1 == cp_action.n0 + 1
            # Do not advance further than the current location of the adjoint
            assert cp_action.n1 <= n - model_r
            # No data for this step is stored
            assert len(data.intersection(range(cp_action.n0, cp_action.n1))) == 0  # noqa: E501

        model_n = cp_action.n1
        model_steps += cp_action.n1 - cp_action.n0
        if store_ics:
            ics.update(range(cp_action.n0, cp_action.n1))
            snapshots[cp_action.n0] = (set(ics), set(data))

        if store_data:
            data.update(range(cp_action.n0, cp_action.n1))
            if cp_action.storage == StorageType.DISK:
                snapshots[cp_action.n0] = (set(ics), set(data))

        if store_ics or store_data:
            # Written data consists of either forward restart or non-linear
            # dependency data, but not both
            assert len(ics) == 0 or len(data) == 0
            assert len(ics) > 0 or len(data) > 0

        # Non-linear dependency data is either not stored, or is stored for a
        # single step
        assert len(data) <= 1

        # The checkpoint location is associated with the earliest step for
        # which data has been stored
        if len(ics) > 0:
            assert cp_action.n0 == min(ics)
        if len(data) > 0:
            assert cp_action.n0 == min(data)

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

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data
        # The checkpoint contains either forward restart or non-linear
        # dependency data, but not both
        assert len(cp[0]) == 0 or len(cp[1]) == 0
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        if len(cp[0]) > 0:
            # Loading a forward restart checkpoint:

            # The checkpoint data is at least two steps away from the current
            # location of the adjoint
            assert cp_action.n < n - model_r - 1
            ics.update(cp[0])
            model_n = cp_action.n

        if len(cp[1]) > 0:
            # Loading a non-linear dependency data checkpoint:

            # The checkpoint data is exactly one step away from the current
            # location of the adjoint
            assert cp_action.n == n - model_r - 1

            data.clear()
            data.update(cp[1])
            model_n = None

    @action.register(Move)
    def action_move(cp_action):
        nonlocal model_n

        # The checkpoint exists
        assert cp_action.n in snapshots
        assert cp_action.from_storage == StorageType.DISK
        cp = snapshots[cp_action.n]

        # The checkpoint contains forward restart data
        assert len(cp[0]) == 0 or len(cp[1]) == 0
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        if len(cp[0]) > 0:
            # The checkpoint data is at least two steps away from the current
            # location of the adjoint
            assert cp_action.n < n - model_r - 1
            ics.update(cp[0])
            model_n = cp_action.n

        if len(cp[1]) > 0:
            # Loading a non-linear dependency data checkpoint:
            # The checkpoint data is exactly one step away from the current
            # location of the adjoint
            assert cp_action.n == n - model_r - 1

            data.clear()
            data.update(cp[1])
            model_n = None

        del snapshots[cp_action.n]

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        # The correct number of adjoint steps has been taken
        assert model_r == n

    for s in S:
        print(f"{n=:d} {s=:d}")

        model_n = 0
        model_r = 0
        model_steps = 0

        store_ics = False
        ics = set()
        store_data = False
        data = set()

        snapshots = {}

        cp_schedule = MixedCheckpointSchedule(n, s, storage=StorageType.DISK)
        assert n == 1 or cp_schedule.uses_storage_type(StorageType.DISK)
        assert cp_schedule.n == 0
        assert cp_schedule.r == 0
        assert cp_schedule.max_n == n

        for _, cp_action in enumerate(cp_schedule):
            action(cp_action)
            cp_action_lists.append(cp_action)
            # The schedule state is consistent with both the forward and
            # adjoint
            assert model_n is None or model_n == cp_schedule.n
            assert model_r == cp_schedule.r

            # Either no data is being stored, or exactly one of forward restart
            # or non-linear dependency data is being stored
            assert not store_ics or not store_data
            assert len(ics) == 0 or len(data) == 0
            # Non-linear dependency data is stored for at most one step
            assert len(data) <= 1
            # Checkpoint storage limits are not exceeded
            assert len(snapshots) <= s

            if isinstance(cp_action, EndReverse):
                break

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
