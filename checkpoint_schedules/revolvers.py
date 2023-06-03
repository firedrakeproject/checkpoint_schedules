#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.

from .schedule import CheckpointSchedule, Forward, Reverse, Transfer,\
    EndForward, EndReverse
from .revolve_sequences import hrevolve

import logging

__all__ = \
    [
        "RevolveCheckpointSchedule"
    ]


class RevolveCheckpointSchedule(CheckpointSchedule):
    """An H-Revolve checkpointing schedule.

    Attributes
    ----------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snapshots_in_ram : int
       The maximum number of forward restart checkpoints to store in memory.
    snapshots_on_disk : int
        The maximum number of forward restart checkpoints to store on disk.
    wvect : tuple, optional
        A two element defining the write cost associated with saving a forward 
        restart checkpoint to RAM (first element) and disk (second element).
    rvect : tuple, optional
        A two element defining the read cost associated with loading a forward 
        restart checkpoint from RAM (first element) and disk (second element).
    uf : float, optional
        The cost of advancing the forward one step.
    ub : float, optional
        The cost of advancing the forward one step, storing non-linear 
        dependency data, and then advancing the adjoint over that step.
    """

    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 wvect=(0.0, 0.1), rvect=(0.0, 0.1), uf=1.0, ub=2.0, **kwargs):
        super().__init__(max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._exhausted = False

        cvect = (snapshots_in_ram, snapshots_on_disk)
        schedule = hrevolve(max_n - 1, cvect, wvect, rvect,
                            uf=uf, ub=ub, **kwargs)
        self._schedule = list(schedule)

    def iter(self):
        """Iterator object of the checkpoint schedules.

        Yields
        ------
        object
            Schedule actions.

        """
        if self._max_n is None:
            raise RuntimeError("Invalid forward steps number.")

        snapshots = set()
        deferred_cp = None
        w_storage = None
        write_ics = False
        write_data = False

        i = 0
        while i < len(self._schedule):
            cp_action, (n_0, n_1, storage) = _convert_action(self._schedule[i])
            if i == 0:
                assert cp_action == "Write"
            if cp_action == "Forward":
                assert i > 0
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                self._n = n_1
                
                w_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i - 1])
                if w_cp_action == "Write":
                    if w_n0 != n_0:
                        raise RuntimeError("Invalid write index.")
                    write_ics = True
                    write_data = False
                    snapshots.add(w_n0)
                else:
                    write_ics = False
                    write_data = True

                yield Forward(n_0, n_1, write_ics, write_data, w_storage)
                if self._n == self._max_n:
                    if self._r != 0:
                        raise RuntimeError("Invalid checkpointing state")
                    yield EndForward()
            elif cp_action == "Backward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                if n_0 != self._max_n - self._r:
                    raise RuntimeError("Invalid checkpointing state")   
                self._r += 1
                yield Reverse(n_0, n_1, clear_fwd_data=True)
                r_cp_action, (r_n0, _, r_storage) = _convert_action(self._schedule[i - 3])
                if r_cp_action == "Read":
                    assert r_n0 == n_1
                    snapshots.remove(n_1)
                    # if r_storage == "RAM":
                    yield Transfer(n_1, r_storage, None, delete=True)
            elif cp_action == "Read":
                if deferred_cp is not None:
                    raise RuntimeError("Invalid checkpointing state")
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])
                if n_cp_action == "Write":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, w_storage)
                elif n_cp_action == "Write_Forward":
                    assert f_cp_action == "Forward"
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, "CHK")
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, "CHK")
                else:
                    raise RuntimeError("Invalid checkpointing state")
            elif cp_action == "Write":
                if n_0 != self._n:
                    raise RuntimeError("Invalid write action index")
            elif cp_action == "Write_Forward":
                if n_0 != self._n + 1:
                    raise RuntimeError("Invalid checkpointing state")
                d_cp_action, (d_n0, _, w_storage) = _convert_action(self._schedule[i + 3])
                if (d_cp_action != "Discard_Forward"
                    or d_n0 != n_0 or w_storage != storage):
                    if w_n0 != n_0:
                        raise RuntimeError("Invalid checkpoint schedule.")
                    write_ics = True
                    write_data = False
            elif cp_action == "Discard":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                # yield Transfer(n_0, storage, None, delete=True)
            elif cp_action == "Discard_Forward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")
            i += 1
     
        if len(snapshots) > 0:
            raise RuntimeError("Unexpected snapshot number.")
        
        self._exhausted = True
        yield EndReverse(True)

    def is_exhausted(self):
        """Exhausted.

        Returns
        -------
        bool
            _description_
        """
        return self._exhausted
    
    def uses_disk_storage(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self._snapshots_on_disk > 0


def _convert_action(action_n):
    """Convert the revolver schedules to ....

    Parameters
    ----------
    action_n : object
        Action object.checkpointing state

    Returns
    -------
    str, tuple(int, int , str)
        Action name, step n_0, step _n1 and storage level.

    """
    cp_action = action_n.type
    if cp_action == "Forward":
        n_0, n_1 = action_n.index
        if n_1 <= n_0:
            raise RuntimeError("Invalid forward index.")
        storage = None
    elif cp_action == "Backward":
        n_0, n_1 = action_n.index
        if n_0 <= n_1:
            raise RuntimeError("Invalid backward indexes.")
        storage = None
    elif cp_action in ["Read", "Write", "Discard",
                       "Write_Forward", "Discard_Forward",
                       "Write_forward_memory",
                       "Discard_forward_memory"]:
        storage, n_0 = action_n.index
        n_1 = None
        storage = {0: "RAM", 1: "disk"}[storage]
    else:
        raise RuntimeError(f"Unexpected action: {cp_action:s}")
    return cp_action, (n_0, n_1, storage)

