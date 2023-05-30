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
    Forward, Reverse, Read, Delete, EndForward, EndReverse
from checkpoint_schedules import \
    (RevolveCheckpointSchedule)

import functools
import pytest


# def memory(n, s):
#     return (MemoryCheckpointSchedule(),
#             {"RAM": 0, "disk": 0}, 1 + n)


# def periodic_disk(n, s, *, period):
#     return (PeriodicDiskCheckpointSchedule(period),
#             {"RAM": 0, "disk": 1 + (n - 1) // period}, period)


# def multistage(n, s):
#     return (MultistageCheckpointSchedule(n, 0, s),
#             {"RAM": 0, "disk": s}, 1)


# def two_level(n, s, *, period):
#     return (TwoLevelCheckpointSchedule(period, s, binomial_storage="RAM"),
#             {"RAM": s, "disk": 1 + (n - 1) // period}, 1)


def h_revolve(n, s):
   
    if s <= 1:
        return (None,
                {"RAM": 0, "disk": 0}, 0)
    else:
        return (RevolveCheckpointSchedule(n, s // 2, s - (s // 2)),
                {"RAM": s // 2, "disk": s - (s // 2)}, 1)


# def mixed(n, s):
#     return (MixedCheckpointSchedule(n, s),
#             {"RAM": 0, "disk": s}, 1)


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
     pytest.param(
         h_revolve, {},
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
def test_validity(schedule, schedule_kwargs,
                  n, S):
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
        # if cp_action.write_ics:
        #     # No forward restart data for these steps is stored
        #     assert len(ics.intersection(range(cp_action.n0, n1))) == 0

        if cp_action.write_data:
            # No non-linear dependency data for these steps is stored
            assert len(data.intersection(range(cp_action.n0, n1))) == 0

        if cp_action.clear:
            ics.clear()
            data.clear()
        if cp_action.write_ics:
            ics.update(range(cp_action.n0, n1))
            snapshots[cp_action.storage][cp_action.n0] = set(ics)
        if cp_action.write_data:
            data.update(range(cp_action.n0, n1))
            fwd_data[cp_action.storage][cp_action.n0] = set(data)

        assert len(ics) > 0 or len(data) > 0
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
        if cp_action.clear_fwd_data:
            data.clear()
            del fwd_data["RAM"][cp_action.n0]
          

    @action.register(Read)
    def action_read(cp_action):
        nonlocal model_n

        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.storage]

        cp = snapshots[cp_action.storage][cp_action.n]

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data

        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp) > 0 or len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r

        model_n = None
       
        if len(cp) > 0:
            ics.clear()
            ics.update(cp)
            model_n = cp_action.n

        # if len(cp[1]) > 0:
        #     data.clear()
        #     data.update(cp[1])

    
    @action.register(Delete)
    def action_delete(cp_action):
        # pass
        nonlocal model_n
        model_n = None
        del snapshots[cp_action.storage][cp_action.n]

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
        # print(f"{n=:d} {s=:d}")
       
        model_n = 0
        model_r = 0
        ics = set()
        data = set()

        snapshots = {"RAM": {}, "disk": {}}
        fwd_data = {"RAM": {}}
        cp_schedule, storage_limits, data_limit = schedule(n, s, **schedule_kwargs)  # noqa: E501
        if cp_schedule is None:
            pytest.skip("Incompatible with schedule type")
        assert cp_schedule.n() == 0
        assert cp_schedule.r() == 0
        assert cp_schedule.max_n() is None or cp_schedule.max_n() == n
        c = 0
        while True:
            cp_action = next(cp_schedule)
            action(cp_action)
            # The schedule state is consistent with both the forward and
            # adjoint
            
            assert model_n is None or model_n == cp_schedule.n()
            assert model_r == cp_schedule.r()

            # Checkpoint storage limits are not exceeded
            for storage_type, storage_limit in storage_limits.items():
                assert len(snapshots[storage_type]) <= storage_limit + 1
            # Data storage limit is not exceeded
            assert min(1, len(ics)) + len(data) <= data_limit
            c += 1
            if isinstance(cp_action, EndReverse):
                break