#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

import functools
import pytest
from checkpoint_schedules.schedule import (
    Forward, Reverse, Copy, Move, EndForward, EndReverse, StorageType)
from checkpoint_schedules import (
    HRevolve, DiskRevolve, PeriodicDiskRevolve, Revolve,
    MultistageCheckpointSchedule, TwoLevelCheckpointSchedule,
    MixedCheckpointSchedule, SingleDiskStorageSchedule,
    SingleMemoryStorageSchedule)


def h_revolve(n, s):
    snap_ram = s // 3
    snap_disk = s - snap_ram
    if snap_ram < 1 or snap_disk < 1:
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        cp_schedule = HRevolve(n, snap_ram, snap_disk)
        return (cp_schedule,
                {StorageType.RAM: snap_ram, StorageType.DISK: snap_disk}, 1)


def disk_revolve(n, s):
    if s < 1:
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        cp_schedule = DiskRevolve(n, s, n - s)
        return (cp_schedule,
                {StorageType.RAM: s, StorageType.DISK: n - s}, 1)


def multistage(n, s):
    return (MultistageCheckpointSchedule(n, 0, s),
            {StorageType.RAM: 0, StorageType.DISK: s}, 1)


def twolevel_binomial(n, s):
    return (TwoLevelCheckpointSchedule(2, s, binomial_storage=StorageType.RAM),
            {StorageType.RAM: s, StorageType.DISK: 1 + (n - 1) // 2}, 1)


def periodic_disk(n, s):
    if s < 1:
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        cp_schedule = PeriodicDiskRevolve(n, s)
        return (cp_schedule,
                {StorageType.RAM: s, StorageType.DISK: n - s}, 1)


def revolve(n, s):
    if s < 1:
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        cp_schedule = Revolve(n, s)
        return (cp_schedule,
                {StorageType.RAM: s, StorageType.DISK: 0}, 1)


def mixed(n, s):
    return (MixedCheckpointSchedule(n, s),
            {StorageType.RAM: 0, StorageType.DISK: s}, 1)


def single_disk_copy(n, s):
    cp_schedule = SingleDiskStorageSchedule(move_data=False)
    return (cp_schedule,
            {StorageType.RAM: 0, StorageType.DISK: n}, 1)


def single_disk_move(n, s):
    cp_schedule = SingleDiskStorageSchedule(move_data=True)
    return (cp_schedule,
            {StorageType.RAM: 0, StorageType.DISK: n}, 1)


def single_memory(n, s):
    cp_schedule = SingleMemoryStorageSchedule()
    return (cp_schedule,
            {StorageType.RAM: 0, StorageType.DISK: 0}, n)


@pytest.mark.parametrize(
    "schedule",
    [
     h_revolve,
     disk_revolve,
     multistage,
     twolevel_binomial,
     periodic_disk,
     revolve,
     mixed,
     single_disk_copy,
     single_disk_move,
     single_memory
     ]
     )
@pytest.mark.parametrize("n, S", [
                                  (1, (0,)),
                                  (2, (1,)),
                                  (3, (1, 2)),
                                  (5, (2,)),
                                  (10, tuple(range(2, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))
                                  ])
def test_validity(schedule, n, S):
    """Test validity of checkpoint schedules. Tests that an adjoint calculation
    can be performed without exceeding storage limits.

    Parameters
    ----------
    schedule : callable
        Accepts the number of forward steps and checkpoint units, and returns a
        schedule.
    n : int
        Number of forward steps.
    S : int
        Number of checkpoint units.
    """

    @functools.singledispatch
    def action(cp_action):
        raise TypeError("Unexpected action")

    @action.register(Forward)
    def action_forward(cp_action):
        nonlocal model_n

        # Start at the current location of the forward
        assert model_n is not None and cp_action.n0 == model_n
        # If the schedule has been finalized, end at or before the end of the
        # forward
        assert cp_schedule.max_n is None or cp_action.n1 <= n

        if cp_schedule.max_n is not None:
            # Do not advance further than the current location of the adjoint
            assert cp_action.n1 <= n - model_r
        n1 = min(cp_action.n1, n)

        model_n = n1
        ics.clear()
        data.clear()
        if cp_action.write_ics:
            cp_ics = set(range(cp_action.n0, n1))
        else:
            cp_ics = set()
        if cp_action.write_adj_deps:
            cp_data = set(range(cp_action.n0, n1))
        else:
            cp_data = set()

        if cp_action.storage in {StorageType.RAM, StorageType.DISK}:
            assert cp_action.n0 not in snapshots[cp_action.storage]
            snapshots[cp_action.storage][cp_action.n0] = (set(cp_ics), set(cp_data))  # noqa: E501
        elif cp_action.storage == StorageType.WORK:
            assert len(ics.intersection(cp_ics)) == 0
            ics.update(cp_ics)
            assert len(data.intersection(cp_data)) == 0
            data.update(cp_data)
        elif cp_action.storage == StorageType.NONE:
            pass
        else:
            raise ValueError("Unexpected storage")

        if len(ics) > 0:
            if len(data) > 0:
                assert cp_action.n0 == min(min(ics), min(data))
            else:
                assert cp_action.n0 == min(ics)
        elif len(data) > 0:
            assert cp_action.n0 == min(data)

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

        # No data is currently stored
        assert len(ics) == 0
        assert len(data) == 0

        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp_ics, cp_data = snapshots[cp_action.from_storage][cp_action.n]

        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp_ics) > 0 or len(cp_data) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r

        if cp_action.to_storage in {StorageType.RAM, StorageType.DISK}:
            assert cp_action.n not in snapshots[cp_action.to_storage]
            snapshots[cp_action.to_storage][cp_action.n] = (set(cp_ics), set(cp_data))  # noqa: E501
        elif cp_action.to_storage == StorageType.WORK:
            if cp_action.n in cp_ics:
                model_n = cp_action.n
            else:
                model_n = None
            assert len(ics.intersection(cp_ics)) == 0
            ics.update(cp_ics)
            if model_n is not None:
                assert ics.issuperset(range(model_n, n - model_r))
            assert len(data.intersection(cp_data)) == 0
            data.update(cp_data)
        elif cp_action.storage == StorageType.NONE:
            pass
        else:
            raise ValueError("Unexpected storage")

    @action.register(Move)
    def action_move(cp_action):
        nonlocal model_n

        # No data is currently stored
        assert len(ics) == 0
        assert len(data) == 0

        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp_ics, cp_data = snapshots[cp_action.from_storage].pop(cp_action.n)

        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp_ics) > 0 or len(cp_data) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r

        if cp_action.to_storage in {StorageType.RAM, StorageType.DISK}:
            assert cp_action.n not in snapshots[cp_action.to_storage]
            snapshots[cp_action.to_storage][cp_action.n] = (set(cp_ics), set(cp_data))  # noqa: E501
        elif cp_action.to_storage == StorageType.WORK:
            if cp_action.n in cp_ics:
                model_n = cp_action.n
            else:
                model_n = None
            assert len(ics.intersection(cp_ics)) == 0
            ics.update(cp_ics)
            if model_n is not None:
                assert ics.issuperset(range(model_n, n - model_r))
            assert len(data.intersection(cp_data)) == 0
            data.update(cp_data)
        elif cp_action.storage == StorageType.NONE:
            pass
        else:
            raise ValueError("Unexpected storage")

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        nonlocal model_r

        # The correct number of adjoint steps has been taken
        assert model_r == n

        if not cp_schedule.is_exhausted:
            model_r = 0

    for s in S:
        print(f"{n=:d} {s=:d}")

        model_n = 0
        model_r = 0
        ics = set()
        data = set()

        snapshots = {StorageType.RAM: {}, StorageType.DISK: {}}
        cp_schedule, storage_limits, data_limit = schedule(n, s)
        if cp_schedule is None:
            pytest.skip("Incompatible with schedule type")
        assert cp_schedule.n == 0
        assert cp_schedule.r == 0
        assert cp_schedule.max_n is None or cp_schedule.max_n == n

        for _, cp_action in enumerate(cp_schedule):
            action(cp_action)

            # The schedule state is consistent with both the forward and
            # adjoint
            assert model_n is None or model_n == cp_schedule.n
            assert model_r == cp_schedule.r
            assert cp_schedule.max_n is None or cp_schedule.max_n == n

            # Checkpoint storage limits are not exceeded
            for storage_type, storage_limit in storage_limits.items():
                assert len(snapshots[storage_type]) <= storage_limit
            # Data storage limit is not exceeded
            assert min(1, len(ics)) + len(data) <= data_limit

            if isinstance(cp_action, EndReverse):
                break
