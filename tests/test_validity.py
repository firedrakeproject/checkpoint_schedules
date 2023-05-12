#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from checkpoint_schedules.schedule import CheckpointSchedule, Clear, Configure, Forward, Reverse, \
    Read, Write, EndForward, EndReverse, WriteForward
from checkpoint_schedules import HRevolveCheckpointSchedule

import functools
import pytest


def h_revolve(n, s, p):
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
        print(p)
        s_disk = s//p
        s_ram = abs(s - s_disk)
        if s_ram == 0:
            s_ram = 1
            s_disk -= 1
        print("s_ram = ", s_ram, "s_disk = ", s_disk)
        assert (s_ram + s_disk) == s
        return (HRevolveCheckpointSchedule(n, s_ram, s_disk),
                {"RAM": s_ram, "disk": s_disk}, 1)
                

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
                                    (25, 5), 
                                    (25, 10), 
                                    (25, 15),
                                    (100, 10), 
                                    (100, 20), 
                                    (100, 30),
                                    (250, 25), 
                                    (250, 50), 
                                    (250, 75),
                                    (250, 100), 
                                    (250, 125),
                                    (250, 150)
                                ])
@pytest.mark.parametrize("p", [2, 3, 5, 4])
def test_validity(schedule, n, S, p):
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
    @functools.singledispatch
    def action(cp_action):
        raise TypeError("Unexpected action")

    @action.register(Clear)
    def action_clear(cp_action):
        if cp_action.clear_ics:
            ics.clear()
        if cp_action.clear_data:
            data.clear()

    @action.register(Configure)
    def action_configure(cp_action):
        nonlocal store_ics, store_data

        store_ics = cp_action.store_ics
        store_data = cp_action.store_data

    @action.register(Write)
    def action_write(cp_action):
        assert len(ics) > 0 and len(sol) > 0
        if len(ics) > 0:
            assert cp_action.n == max(ics)
        snapshots[cp_action.storage][cp_action.n] = (set(ics), set(sol))

    @action.register(WriteForward)
    def action_write_forward(cp_action):
        assert len(ics) == 0 and len(data) > 0
        assert cp_action.n == max(data)
        assert data == cp_schedule.forward_data
        fwd_chk['RAM'][cp_action.n] = (set(data), set(sol))

    @action.register(Forward)
    def action_forward(cp_action):
        nonlocal model_n
        nonlocal sol
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

    @action.register(Reverse)
    def action_reverse(cp_action):
        nonlocal model_r
        # Start at the current location of the adjoint
        assert cp_action.n1 == n - model_r
        # Advance at least one step
        assert cp_action.n0 < cp_action.n1
        # Non-linear dependency data for these steps is stored
        sol.clear()
        
        model_r += cp_action.n1 - cp_action.n0
        if len(cp_schedule.forward_data) == 0:
            fwd_chk["RAM"].clear()

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
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        # print(cp_action.n, n, model_r)
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

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        nonlocal model_r

        assert model_r == cp_schedule.max_n()
        assert len(data) == 0
        if not cp_action.exhausted:
            model_r = 0
   
    # for s in S:
    s=S
    # for p0 in p:
    p0 = p
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

    cp_schedule, storage_limits, data_limit = schedule(n, s, p0)  # noqa: E501
    if cp_schedule is None:
        pytest.skip("Incompatible with schedule type")
    assert cp_schedule.n() == 0
    assert cp_schedule.r() == 0
    assert s <= n
    assert cp_schedule.max_n() is None or cp_schedule.max_n() == n
    assert n >= 0

    def initial_condition(init_condition, sol, ics):
        """Set the initial condition.
        """
        nonlocal model_n
        sol.add(init_condition)
        ics.add(model_n)

    initial_condition(init_condition, sol, ics)
    c = 0
    while True:
        cp_action = next(cp_schedule) 
        # print(cp_action, c)       
        action(cp_action)    

        assert model_n is None or model_n == cp_schedule.n()
        assert model_r == cp_schedule.r()
        # Checkpoint storage limits are not exceeded
        for storage_type, storage_limit in storage_limits.items():
            assert len(snapshots[storage_type]) <= storage_limit
            
        # Data storage limit is not exceeded
        assert min(1, len(ics)) + len(data) <= data_limit
        c += 1
        if isinstance(cp_action, EndReverse):
            break
