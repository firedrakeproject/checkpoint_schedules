#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from checkpoint_schedules import HRevolveCheckpointSchedule

import functools
import pytest


def h_revolve(n, s):
    """Return H-Revolve sequence.

    Parameters
    ----------
    n : int
        Number of forward step to execute in the AC graph.
    s : tuple
        The number of slots in each level of memory.

    Returns
    -------
    object
        H-Revolve generator.
    """
    if s <= 1:
        return (None,
                {"RAM": 0, "disk": 0}, 0)
    else:
        return (HRevolveCheckpointSchedule(n, s // 2, s - (s // 2)),
                {"RAM": s // 2, "disk": s - (s // 2)}, 1)


@pytest.mark.parametrize(
    "schedule",
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
            h_revolve),
        #  (mixed, {})
    ])
@pytest.mark.parametrize("n, S", [
                                # (1, (0,)),
                                #   (2, (1,)),
                                #   (3, (1, 2)),
                                #   (10, tuple(range(1, 10))),
                                #   (100, tuple(range(1, 100))),
                                  (100, tuple(range(10, 100, 10)))])
def test_validity(schedule, n, S):
    """Test validity.

    Parameters
    ----------
    schedule : object
        Scheduler object.
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
        init_condition = 0
        store_ics = False
        store_data = False
        ics = set()
        data = set()
        sol = set()
        snapshots = {"RAM": {}, "disk": {}}
        fwd_chk = {"RAM": {}}
       
        cp_schedule, storage_limits, data_limit = schedule(n, s)  # noqa: E501
        print(cp_schedule._schedule)
        if cp_schedule is None:
            pytest.skip("Incompatible with schedule type")
        assert cp_schedule.n() == 0
        assert cp_schedule.r() == 0
        assert cp_schedule.max_n() is None or cp_schedule.max_n() == n
        
        def initial_condition():
            """Set the initial condition.
            """
            sol.add(init_condition)
            ics.add(model_n)

        initial_condition()
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
                assert len(ics) > 0 and len(sol) > 0
                if len(ics) > 0:
                    assert cp_action.n == max(ics)
                snapshots[cp_action.storage][cp_action.n] = (set(ics), set(sol))
            elif cp_action.type == "WriteForward":
                assert len(ics) == 0 and len(data) > 0
                assert cp_action.n == max(data)
                fwd_chk['RAM'][cp_action.n] = (set(data), set(sol))
            elif cp_action.type == "Forward":
                assert model_n is not None and model_n == cp_action.n0
                assert cp_action.n0 < cp_action.n1
                assert len(sol) == 1 and cp_action.n0 in sol
                if cp_schedule.max_n() is not None:
                    # Do not advance further than the current location of the adjoint
                    assert cp_action.n1 <= n - model_r
               
                n1 = min(cp_action.n1, n)
                if store_ics:
                    # No forward restart data for these steps is stored
                    assert n1 not in snapshots["RAM"] and n1 not in snapshots["disk"]
                if store_data:
                    # No non-linear dependency data for these steps is stored
                    assert n1 not in fwd_chk['RAM']
                model_n = n1
                if store_ics:
                    ics.add(n1)
                if store_data:
                    data.add(n1)
                if n1 == n:
                    cp_schedule.finalize(n1)
                sol = set()
                sol.add(n1)
            elif cp_action.type == "Read":
                # The checkpoint exists
                assert cp_action.n in snapshots[cp_action.storage]
                cp = snapshots[cp_action.storage][cp_action.n]

                # No data is currently stored for this step
                assert cp_action.n not in ics
        
                # The checkpoint contains forward restart or non-linear dependency data
                assert len(cp[0]) > 0 or len(cp[1]) > 0
    
                # The checkpoint data is before the current location of the adjoint
                assert cp_action.n <= n - model_r

                model_n = None

                if len(cp[0]) > 0:
                    ics.clear()
                    ics.update(cp[0])
                    model_n = cp_action.n

                if len(cp[1]) > 0:
                    sol.clear()
                    sol.update(cp[1])

                if cp_action.delete:
                    del snapshots[cp_action.storage][cp_action.n]

            elif cp_action.type == "Reverse":
                # Start at the current location of the adjoint
                assert cp_action.n1 == n - model_r
                # Advance at least one step
                assert cp_action.n0 < cp_action.n1
                # Non-linear dependency data for these steps is stored
                sol.clear()

                model_r += cp_action.n1 - cp_action.n0

            elif cp_action.type == "EndForward":
                assert model_n is not None and model_n == cp_schedule.max_n()
            elif cp_action.type == "EndReverse":
                assert model_r == cp_schedule.max_n()
                assert len(data) == 0
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

            if cp_action.type == "EndReverse":
                break
