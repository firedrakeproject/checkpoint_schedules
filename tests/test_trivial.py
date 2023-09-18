#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
import pytest
from checkpoint_schedules.schedule import \
    Forward, Reverse, Copy, Move, EndForward, EndReverse, StorageType
from checkpoint_schedules import SingleDiskStorageSchedule, \
    SingleMemoryStorageSchedule


def single_disk_copy(n):
    checkpointing = SingleDiskStorageSchedule()
    return (checkpointing,
            {StorageType.RAM: 0, StorageType.DISK: n}, 1)


def single_disk_move(n):
    checkpointing = SingleDiskStorageSchedule(move_data=True)
    return (checkpointing,
            {StorageType.RAM: 0, StorageType.DISK: n}, 1)


def single_memory(n):
    checkpointing = SingleMemoryStorageSchedule()
    return (checkpointing,
            {StorageType.RAM: 0, StorageType.DISK: 0}, n)


@pytest.mark.parametrize(
    "schedule",
    [
     single_disk_copy,
     single_disk_move,
     single_memory,
     ]
     )


def test_validity(schedule, n=10):
    """Test the checkpoint revolvers.

    Parameters
    ----------
    schedule : object
        Revolver schedule.
    n : int
        Total forward steps.
    S : int
        Snapshots.
    """
    @functools.singledispatch
    def action(cp_action):
        raise TypeError("Unexpected action")

    @action.register(Forward)
    def action_forward(cp_action):
        nonlocal model_n
        # Start at the current location of the forward
        assert model_n is not None and model_n == cp_action.n0

        assert cp_action.storage == StorageType.WORK or \
            cp_action.storage == StorageType.DISK
        # If the schedule has been finalized, end at or before the end of the
        # forward
        assert cp_schedule.max_n is None or cp_action.n1 <= n
        if cp_schedule.max_n is not None:
            # Do not advance further than the current location of the adjoint
            assert cp_action.n1 <= n - model_r

        n1 = min(cp_action.n1, n)

        model_n = n1
        data.clear()
        assert not cp_action.write_ics
        assert cp_action.write_adj_deps is True
        assert len(data.intersection(range(cp_action.n0, n1))) == 0
        if cp_action.storage == StorageType.DISK:
            for step in range(cp_action.n0, n1):
                data.update(range(step, step + 1))
                snapshots[cp_action.storage][step] = (set(ics), set(data))
                data.clear()
            assert cp_action.n0 == min(snapshots[cp_action.storage])
        else:
            data.update(range(cp_action.n0, n1))
            assert cp_action.n0 == min(data)

        assert len(ics) == 0

        if n1 == n:
            cp_schedule.finalize(n1)

    @action.register(Reverse)
    def action_reverse(cp_action):
        nonlocal model_r

        # Start at the current location of the adjoint
        assert cp_action.n1 == n - model_r
        # Advance at least one step
        assert cp_action.n0 < cp_action.n1
        # Non-linear dependency data for these steps is stored
        assert data.issuperset(range(cp_action.n0, cp_action.n1))

        model_r += cp_action.n1 - cp_action.n0
        if cp_action.clear_adj_deps:
            data.clear()

    @action.register(Copy)
    def action_copy(cp_action):
        nonlocal model_n
        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp = snapshots[cp_action.from_storage][cp_action.n]

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data

        # The checkpoint contains forward data
        assert len(cp[0]) == 0 and len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r
        model_n = None

        data.clear()
        data.update(cp[1])

    @action.register(Move)
    def action_move(cp_action):
        nonlocal model_n
        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.from_storage]
        assert cp_action.n == max(snapshots[cp_action.from_storage])
        cp = snapshots[cp_action.from_storage][cp_action.n]

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data

        # The checkpoint contains forward data
        assert len(cp[0]) == 0 and len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r
        model_n = None
        data.clear()
        data.update(cp[1])
        del snapshots[cp_action.from_storage][cp_action.n]

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        nonlocal model_r, cp_schedule

        # The correct number of adjoint steps has been taken
        assert model_r == n
        is_exhausted = cp_schedule.is_exhausted
        if is_exhausted is False:
            model_r = 0

    model_n = 0
    model_r = 0
    ics = set()
    data = set()
    assert isinstance(n, int)
    snapshots = {StorageType.RAM: {}, StorageType.DISK: {}}
    cp_schedule, storage_limits, data_limit = schedule(n)
    if cp_schedule is None:
        raise TypeError("Incompatible with schedule type.")
    assert cp_schedule.n == 0
    assert cp_schedule.r == 0
    assert cp_schedule.max_n is None or cp_schedule.max_n == n

    for _, cp_action in enumerate(cp_schedule):
        action(cp_action)
        assert model_n is None or model_n == cp_schedule.n
        assert model_r == cp_schedule.r

        # Checkpoint storage limits are not exceeded
        for storage_type, storage_limit in storage_limits.items():
            assert len(snapshots[storage_type]) <= storage_limit
        # Data storage limit is not exceeded
        assert min(1, len(ics)) + len(data) <= data_limit

        if isinstance(cp_action, EndReverse):
            break
