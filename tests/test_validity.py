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
    Forward, Reverse, Copy, EndForward, EndReverse, StorageLevel
from checkpoint_schedules import HRevolve, DiskRevolve, PeriodicDiskRevolve

import functools
import pytest

def h_revolve(n, s):
    """_summary_

    Parameters
    ----------
    n : _type_
        _description_
    s : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
   
    if s < 1:
        return (None,
                {StorageLevel(0).name: 0, StorageLevel(1).name: 0}, 0)
    else:
        revolver = HRevolve(n, s, n - s)
        revolver.sequence(w_cost=(0, 2.0), r_cost=(0, 2.0))
        return (revolver,
                {StorageLevel(0).name: s, StorageLevel(1).name: n - s}, 1)


def disk_revolve(n, s):
    """_summary_

    Parameters
    ----------
    n : _type_
        _description_
    s : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if s <= 1:
        return (None,
                {StorageLevel(0).name: 0, StorageLevel(1).name: 0}, 0)
    else:
        revolver = DiskRevolve(n, s, n - s)
        revolver.sequence()
        return (revolver,
                {StorageLevel(0).name: s, StorageLevel(1).name: n - s}, 1)


def periodic_disk(n, s, period):
    """_summary_

    Parameters
    ----------
    n : _type_
        _description_
    s : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if s < 1:
        return (None,
                {StorageLevel(0).name: 0, StorageLevel(1).name: 0}, 0)
    else:
        print(n, s)
        revolver = PeriodicDiskRevolve(n, s, n)
        revolver.sequence(period=period)
        
        return (revolver,
                {StorageLevel(0).name:  s, StorageLevel(1).name: n - s}, 1)

@pytest.mark.parametrize(
    "schedule, schedule_kwargs",
    [
     (periodic_disk, {"period": 2}),
     (periodic_disk, {"period": 4}),
     (periodic_disk, {"period": 8}),
     (periodic_disk, {"period": 16}),
     (disk_revolve, {}),
     (h_revolve, {})
     ]
     )
@pytest.mark.parametrize("n, S", [
                                #   (1, (0,)),
                                #   (5, (2,)),
                                #   (3, (1, 2)),
                                #   (10, tuple(range(1, 10))),
                                #   (100, tuple(range(1, 100))),
                                  (100, tuple(range(10, 70, 10)))
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
        """_summary_

        Parameters
        ----------
        cp_action : _type_
            _description_
        """
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
            assert cp_action.n0 not in snapshots[cp_action.storage]
            # len(ics.intersection(range(cp_action.n0, n1))) == 0

        if cp_action.write_adj_deps:
            # No non-linear dependency data for these steps is stored
            assert len(data.intersection(range(cp_action.n0, n1))) == 0

        ics.clear()
        data.clear()
        if cp_action.write_ics:
            ics.update(range(cp_action.n0, n1))
            snapshots[cp_action.storage][cp_action.n0] = (set(ics), set(data))
        if cp_action.write_adj_deps:
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
        """_summary_

        Parameters
        ----------
        cp_action : _type_
            _description_
        """
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
        model_n = None
        assert cp_action.n in snapshots[cp_action.from_storage]
        cp = snapshots[cp_action.from_storage][cp_action.n]
        assert cp_action.n not in ics
        assert cp_action.n not in data
        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp[0]) > 0 or len(cp[1]) > 0
        assert cp_action.n < n - model_r
        if len(cp[0]) > 0:
            ics.clear()
            ics.update(cp[0])
            model_n = cp_action.n

        if len(cp[1]) > 0:
            data.clear()
            data.update(cp[1])
        if cp_action.delete:
            del snapshots[cp_action.from_storage][cp_action.n]

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

        snapshots = {StorageLevel(0).name: {}, StorageLevel(1).name: {}}
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


