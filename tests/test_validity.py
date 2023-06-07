#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from checkpoint_schedules.schedule import \
    Forward, Reverse, Transfer, EndForward, EndReverse
from checkpoint_schedules import RevolveCheckpointSchedule, StorageLocation

import functools
import pytest


# def memory(n, s):
#     return (MemoryCheckpointSchedule(),
#             {StorageLocation(0).name: 0, StorageLocation(1).name: 0}, 1 + n)


# def periodic_disk(n, s, *, period):
#     return (PeriodicDiskCheckpointSchedule(period),
#             {StorageLocation(0).name: 0, StorageLocation(1).name: 1 + (n - 1) // period}, period)


# def multistage(n, s):
#     return (MultistageCheckpointSchedule(n, 0, s),
#             {StorageLocation(0).name: 0, StorageLocation(1).name: s}, 1)


# def two_level(n, s, *, period):
#     return (TwoLevelCheckpointSchedule(period, s, binomial_storage=StorageLocation(0).name),
#             {StorageLocation(0).name: s, StorageLocation(1).name: 1 + (n - 1) // period}, 1)


def h_revolve(n, s):
   
    if s <= 1:
        return (None,
                {StorageLocation(0).name: 0, StorageLocation(1).name: 0}, 0)
    else:
        return (RevolveCheckpointSchedule(n, s, 0),
                {StorageLocation(0).name: s, StorageLocation(1).name: 0}, 1)


def disk_revolve(n, s):
    if s <= 1:
        return (None,
                {StorageLocation(0).name: 0, StorageLocation(1).name: 0}, 0)
    else:
        return (RevolveCheckpointSchedule(n, s, schedule="disk_revolve"),
                {StorageLocation(0).name: s, StorageLocation(1).name: n - s}, 1)

def periodic_disk_revolve(n, s):
    if s <= 1:
        return (None,
                {StorageLocation(0).name: 0, StorageLocation(1).name: 0}, 0)
    else:
        return (RevolveCheckpointSchedule(n, s, schedule="periodic_disk_revolve"),
                {StorageLocation(0).name: s, StorageLocation(1).name: n - s}, 1)
# def mixed(n, s):
#     return (MixedCheckpointSchedule(n, s),
#             {StorageLocation(0).name: 0, StorageLocation(1).name: s}, 1)


@pytest.mark.parametrize(
    "schedule, schedule_kwargs",
    [
    # (memory, {}),
    #  (periodic_disk, {"period": 1}),
    #  (periodic_disk, {"period": 2}),
    #  (periodic_disk, {"period": 7}),
    #  (periodic_disk, {"period": 10}),
    #  (multistage, {}),
    #  (two_level, {"period": 1}),
    #  (two_level, {"period": 2}),
    #  (two_level, {"period": 7}),
    #  (two_level, {"period": 10}),
    
        (h_revolve, {},
        #  marks=pytest.mark.skipif(hrevolve is None,
        #                           reason="H-Revolve not available")),
    #  (mixed, {}
    )
     ]
     )
@pytest.mark.parametrize("n, S", [
                                #   (1, (0,)),
                                #   (5, (2,)),
                                #   (3, (1, 2)),
                                #   (10, tuple(range(1, 10))),
                                #   (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))
                                  ])
def test_validity(schedule, schedule_kwargs, n, S):
    """Test the checkpoint revolvers.

    Parameters
    ----------
    schedule : object
        Revolver schedule.
    schedule_kwargs : _type_
        _description_
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

        if cp_schedule.max_n() is not None:
            # Do not advance further than the current location of the adjoint
            assert cp_action.n1 <= n - model_r
        n1 = min(cp_action.n1, n)
        
        model_n = n1
        if cp_action.write_ics:
            # No forward restart data for these steps is stored
            assert len(ics.intersection(range(cp_action.n0, n1))) == 0

        if cp_action.adj_deps:
            # No non-linear dependency data for these steps is stored
            assert len(data.intersection(range(cp_action.n0, n1))) == 0

        ics.clear()
        data.clear()
        if cp_action.write_ics:
            ics.update(range(cp_action.n0, n1))
            snapshots[cp_action.storage][cp_action.n0] = (set(ics), set(data))
        if cp_action.adj_deps:
            data.update(range(cp_action.n0, n1))

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
    
    @action.register(Transfer)
    def action_transfer(cp_action):
        nonlocal model_n
        model_n = None
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp = snapshots[cp_action.from_storage][cp_action.n]
        assert cp_action.n not in ics
        assert cp_action.n not in data
        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp[0]) > 0 or len(cp[1]) > 0
        
        if cp_action.delete:
            assert cp_action.n == n - model_r
            del snapshots[cp_action.from_storage][cp_action.n]
        elif cp_action.to_storage == StorageLocation(0).name:
            # No data is currently stored for this step
            assert cp_action.n < n - model_r
            if cp_action.from_storage == StorageLocation(1).name:
                assert cp_action.to_storage == StorageLocation(0).name
            ics.clear()
            ics.update(cp[0])
            model_n = cp_action.n
        else:
            assert cp_action.n < n - model_r
            data.update(cp[1])
            model_n = cp_action.n
            # fwd_data[cp_action.to_storage][cp_action.n] = set(data)

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        nonlocal model_r

        # The correct number of adjoint steps has been taken
        assert model_r == n

        if not cp_action.exhausted:
            model_r = 0

    for s in S:
        print(f"{n=:d} {s=:d}")
       
        model_n = 0
        model_r = 0
        ics = set()
        data = set()

        snapshots = {StorageLocation(0).name: {}, StorageLocation(1).name: {}}
        cp_schedule, storage_limits, data_limit = schedule(n, s, **schedule_kwargs) 

        if cp_schedule is None:
            raise TypeError("Incompatible with schedule type.")
        assert cp_schedule.n() == 0
        assert cp_schedule.r() == 0
        assert cp_schedule.max_n() is None or cp_schedule.max_n() == n
        while True:
            cp_action = next(cp_schedule)
            action(cp_action)
            assert model_n is None or model_n == cp_schedule.n()
            assert model_r == cp_schedule.r()

            # Checkpoint storage limits are not exceeded
            for storage_type, storage_limit in storage_limits.items():
                assert len(snapshots[storage_type]) <= storage_limit
            # Data storage limit is not exceeded
            assert min(1, len(ics)) + len(data) <= data_limit

            if isinstance(cp_action, EndReverse):
                break