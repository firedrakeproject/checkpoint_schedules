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
from checkpoint_schedules.schedule import \
    Forward, Reverse, Copy, Move, EndForward, EndReverse, StorageType
from checkpoint_schedules import HRevolve, DiskRevolve, PeriodicDiskRevolve, \
    Revolve, MultistageCheckpointSchedule, TwoLevelCheckpointSchedule, \
    MixedCheckpointSchedule


def h_revolve(n, s):
    snap_ram = s//3
    snap_disk = s - s//3
    if s//3 < 1:
        pytest.skip("H-Revolve accepts snapshots in RAM > 1")
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        revolver = HRevolve(n, snap_ram, snap_disk)
        return (revolver,
                {StorageType.RAM: snap_ram, StorageType.DISK: snap_disk}, 1)


def disk_revolve(n, s):
    if s < 1:
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        revolver = DiskRevolve(n, s, n - s)
        return (revolver,
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
        revolver = PeriodicDiskRevolve(n, s)
        return (revolver,
                {StorageType.RAM:  s, StorageType.DISK: n - s}, 1)


def revolve(n, s):
    if s < 1:
        return (None,
                {StorageType.RAM: 0, StorageType.DISK: 0}, 0)
    else:
        revolver = Revolve(n, s)
        return (revolver,
                {StorageType.RAM:  s, StorageType.DISK: 0}, 1)


def mixed(n, s):
    return (MixedCheckpointSchedule(n, s),
            {StorageType.RAM: 0, StorageType.DISK: s}, 1)


@pytest.mark.parametrize(
    "schedule",
    [
     revolve,
     periodic_disk,
     disk_revolve,
     h_revolve,
     multistage,
     twolevel_binomial,
     mixed,
     ]
     )
@pytest.mark.parametrize("n, S", [
                                  (5, (2,)),
                                  (3, (1, 2)),
                                  (10, tuple(range(2, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))
                                  ])
def test_validity(schedule, n, S):
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
            # No forward restart data for these steps is stored
            assert cp_action.n0 not in snapshots[cp_action.storage]
            # No forward restart data for these steps is stored
            assert len(ics.intersection(range(cp_action.n0, n1))) == 0
            ics.update(range(cp_action.n0, n1))
            snapshots[cp_action.storage][cp_action.n0] = (set(ics), set(data))

        if cp_action.write_adj_deps:
            # No non-linear dependency data for these steps is stored
            assert len(data.intersection(range(cp_action.n0, n1))) == 0
            data.update(range(cp_action.n0, n1))
            if cp_action.storage == StorageType.DISK:
                snapshots[cp_action.storage][cp_action.n0] = (set(ics), set(data))  # noqa: E501

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
        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp = snapshots[cp_action.from_storage][cp_action.n]

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data

        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r
        model_n = None
        if len(cp[0]) > 0:
            ics.clear()
            ics.update(cp[0])
            model_n = cp_action.n

        if len(cp[1]) > 0:
            data.clear()
            data.update(cp[1])

    @action.register(Move)
    def action_move(cp_action):
        nonlocal model_n
        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp = snapshots[cp_action.from_storage][cp_action.n]
        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r

        assert cp_action.n < n - model_r

        model_n = None
        if len(cp[0]) > 0:
            ics.clear()
            ics.update(cp[0])
            model_n = cp_action.n

        if len(cp[1]) > 0:
            data.clear()
            data.update(cp[1])

        del snapshots[cp_action.from_storage][cp_action.n]

    @action.register(EndForward)
    def action_end_forward(cp_action):
        ics.clear()
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

    for s in S:
        print(f"{n=:d} {s=:d}")
        model_n = 0
        model_r = 0
        ics = set()
        data = set()

        snapshots = {StorageType.RAM: {}, StorageType.DISK: {}}
        cp_schedule, storage_limits, data_limit = schedule(n, s)
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
