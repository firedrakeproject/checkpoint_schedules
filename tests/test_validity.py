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

from checkpoint_schedules import \
    Clear, Configure, Forward, Reverse, Read, Write, EndForward, EndReverse
from checkpoint_schedules import HRevolveCheckpointSchedule

import functools
import pytest


def h_revolve(n, s):
    """Return H-Revolve sequence.

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
                {"RAM": 0, "disk": 0}, 0)
    else:
        return (HRevolveCheckpointSchedule(n, s // 2, s - (s // 2)),
                {"RAM": s // 2, "disk": s - (s // 2)}, 1)


# def mixed(n, s):
#     return (MixedCheckpointSchedule(n, s),
#             {"RAM": 0, "disk": s}, 1)


@pytest.mark.parametrize(
    "schedule, schedule_kwargs",
    [
        #   (memory, {}),
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
            h_revolve, {},),
        #  (mixed, {})
    ])
@pytest.mark.parametrize("n, S", [(1, (0,)),
                                  (2, (1,)),
                                  (3, (1, 2)),
                                  (10, tuple(range(1, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))])
def test_validity(schedule, schedule_kwargs, n, S):
    """Test validity.

    Parameters
    ----------
    schedule : object
        Scheduler object.
    schedule_kwargs : _type_
        _description_
    n : int
        Number of forward step to execute in the AC graph.
    S : tuple
        The number of slots in each level of memory.

    Raises
    ------
    TypeError
        Unexpected action.
    """

    for s in S:
        print(f"{n=:d} {s=:d}")

        model_n = 0
        model_r = 0

        store_ics = False
        ics = set()
        lfwd = set()
        store_data = False
        data = set()

        snapshots = {"RAM": {}, "disk": {}}
        fwd_data = {"RAM": {}, "disk": {}}

        cp_schedule, storage_limits, data_limit = schedule(n, s, **schedule_kwargs)  # noqa: E501
        if cp_schedule is None:
            pytest.skip("Incompatible with schedule type")
        assert cp_schedule.n() == 0
        assert cp_schedule.r() == 0
        assert cp_schedule.max_n() is None or cp_schedule.max_n() == n

        while True:
            cp_action = next(cp_schedule)
            if cp_action.type == "Clear":
                if cp_action.clear_ics:
                    ics.clear()
                if cp_action.clear_data:
                    data.clear()
            elif cp_action.type == "Configure":
                store_ics = cp_action.store_ics
                store_data = cp_action.store_data
            elif cp_action.type == "Write":
                if store_ics:
                    ics.add(model_n)
                assert ics is not None
                snapshots[cp_action.storage][cp_action.n] = (set(ics), set(data))
            elif cp_action.type == "WriteForward":
                if store_data:
                    lfwd.add(model_n)
                assert lfwd is not None
                fwd_data[cp_action.storage][cp_action.n] = (set(lfwd), set(data))
            elif cp_action.type == "Forward":
                assert len(ics) == 1
                n1 = min(cp_action.n1, cp_schedule.max_n())
                model_n = n1
                if n1 == cp_schedule.max_n():
                    cp_schedule.finalize(n1)
                data.add(model_n)
            elif cp_action.type == "Read":
                cp = snapshots[cp_action.storage][cp_action.n]

                # No data is currently stored for this step
                assert cp_action.n not in ics
                assert cp_action.n not in data
                
                # The checkpoint contains forward restart or non-linear dependency data
                assert len(cp[0]) > 0 or len(cp[1]) > 0

                # The checkpoint data is before the current location of the adjoint
                assert cp_action.n < n - model_r
                model_n = None
                if len(cp) > 0:
                    ics.clear()
                    ics.update(cp)
                if len(data) > 0:
                    data.clear()
                    data.update(cp)
                model_n = cp_action.n
                if cp_action.delete:
                    del snapshots[cp_action.storage][cp_action.n]
            elif cp_action.type == "Reverse":
                assert len(lfwd) == 1 or len(ics) == 1
                if len(lfwd) == 1:
                    assert len(ics) == 0
                if len(ics) == 1:
                    assert len(lfwd) == 0
                
                # Start at the current location of the adjoint
                assert cp_action.n1 == n - model_r
                # Advance at least one step
                assert cp_action.n0 < cp_action.n1
                # Non-linear dependency data for these steps is stored
                # assert data.issuperset(range(cp_action.n0, cp_action.n1))

                model_r += cp_action.n1 - cp_action.n0
                if cp_action.delete:
                    del fwd_data[cp_action.storage][cp_action.n]
            elif cp_action.type == "EndForward":
                assert model_n is not None and model_n == cp_schedule.max_n()
            elif cp_action.type == "EndReverse":
                assert model_r == cp_schedule.max_n()
                assert len(lfwd) == 0
                if not cp_action.exhausted:
                    model_r = 0       
            # The schedule state is consistent with both the forward and
            # adjoint
            assert model_n is None or model_n == cp_schedule.n()
            assert model_r == cp_schedule.r()

            # Checkpoint storage limits are not exceeded
            for storage_type, storage_limit in storage_limits.items():
                assert len(snapshots[storage_type]) <= storage_limit
            # Data storage limit is not exceeded
            assert min(1, len(ics)) + len(data) <= data_limit

            if isinstance(cp_action, EndReverse):
                break
