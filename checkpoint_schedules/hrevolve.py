#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add corect license text
from .schedule import CheckpointSchedule, Clear, Configure, Forward, Reverse, \
    Read, Write, EndForward, EndReverse
from .hrevolve_sequence import hrevolve
import logging

__all__ = \
    [
        "HRevolveCheckpointSchedule"
    ]


class HRevolveCheckpointSchedule(CheckpointSchedule):
    """H-Revolve Checnkpointing Schedule.

    Attributes
    ----------
    max_n : int
        Total checkpoint of a foward solver.
    snapshots_in_ram : int
        Number of checkpoints save in RAM.
    snapshots_on_disk : int
        Number of checkpoints save in disk.
    wvect : tuple, optional
        _description_
    rvect : tuple, optional
        _description_
    cfwd : float, optional
        _description_
    cbwd : float, optional
        _description_
    """
    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 wvect=(0.0, 0.1), rvect=(0.0, 0.1), cfwd=1.0, cbwd=2.0, **kwargs):
        
        super().__init__(max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._exhausted = False
        self.end_forward = (False, None)
        cvect = (snapshots_in_ram, snapshots_on_disk)
        schedule = hrevolve(max_n, cvect, wvect, rvect,
                            cfwd=cfwd, cbwd=cbwd, **kwargs)
        
        self._schedule = list(schedule)

    def iter(self):
        def action(i):
            assert i >= 0 and i < len(self._schedule)
            action = self._schedule[i]
            cp_action = action.type
            if cp_action == "Forward":
                n_0, n_1 = action.index
                if n_1 <= n_0:
                    raise RuntimeError("Invalid schedule")
                storage = None
            elif cp_action == "Backward":
                n_0 = action.index
                n_1 = None
                storage = None
            elif cp_action in ["Read", "Write",
                               "Write_Forward", "Discard",
                               "Discard_Forward"]:
                storage, n_0 = action.index
                n_1 = None
                storage = {0: "RAM", 1: "disk"}[storage]
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")
            return cp_action, (n_0, n_1, storage)

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")

        snapshots = set()
        forward_data = set()
        deferred_cp = None

        def write_deferred_cp():
            nonlocal deferred_cp

            if deferred_cp is not None:
                snapshots.add(deferred_cp[0])
                yield Write(*deferred_cp)
                deferred_cp = None

        for i in range(len(self._schedule)-2):
            cp_action, (n_0, n_1, storage) = action(i)
           
            if cp_action == "Forward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                self._n = n_1
                yield Clear(True, True)
                d_cp_action, (d_n_0, _, d_storage) = action(i + 1)
                if d_cp_action == "Write_Forward":
                    forward_data.add(n_1)
                    yield Configure(n_0 not in snapshots, True)
                else:
                    yield Configure(n_0 not in snapshots, False)
                yield Forward(n_0, n_1)

                if self._n == self._max_n:
                    if self._r != 0:
                        raise RuntimeError("Invalid checkpointing state")
                    yield EndForward()
                    
            elif cp_action == "Backward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                if n_0 != self._max_n - self._r:
                    raise RuntimeError("Invalid checkpointing state")

                yield from write_deferred_cp()

                self._n = n_0
                self._r += 1
                yield Reverse(n_0, n_0-1)
                d_cp_action, (d_n_0, _, d_storage) = action(i + 1)
                if d_cp_action == "Discard_Forward":
                    forward_data.remove(n_0)
            elif cp_action == "Read":
                if deferred_cp is not None:
                    raise RuntimeError("Invalid checkpointing state")

                if n_0 == self._max_n - self._r:
                    cp_delete = True
            
                d_cp_action, (d_n_0, _, d_storage) = action(i + 2)
                if d_cp_action == "Discard":
                    if d_n_0 != n_0 or d_storage != storage:
                        raise RuntimeError("Invalid schedule")
                    cp_delete = True
                else:
                    cp_delete = False

                yield Clear(True, True)
                if cp_delete:
                    snapshots.remove(n_0)
                self._n = n_0
                yield Read(n_0, storage, cp_delete)
            elif cp_action == "Write":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                
                deferred_cp = (n_0, storage)
                yield from write_deferred_cp()

                if i > 0:
                    r_cp_action, (r_n_0, _, _) = action(i - 1)
                    if r_cp_action == "Read":
                        if r_n_0 != n_0:
                            raise RuntimeError("Invalid schedule")
                        yield from write_deferred_cp()
            elif cp_action == "Write_Forward":
                d_cp_action, (d_n_0, _, d_storage) = action(i + 2)
                if d_cp_action != "Discard_Forward":
                    raise RuntimeError("Invalid checkpointing state")
            elif cp_action == "Discard":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                r_cp_action, (r_n_0, _, r_storage) = action(i - 2)
                if r_cp_action != "Read" \
                        or r_n_0 != n_0 \
                        or r_storage != storage:
                    raise RuntimeError("Invalid schedule")
            elif cp_action == "Discard_Forward":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                r_cp_action, (df_n_0, _, df_storage) = action(i - 2)
                if r_cp_action != "Write_Forward" \
                        or df_n_0 != n_0 \
                        or r_storage != storage:
                    raise RuntimeError("Invalid schedule")
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")

        if len(snapshots) != 0:
            raise RuntimeError("Invalid checkpointing state")

        yield Clear(True, True)

        self._exhausted = True
        yield EndReverse(True)

    def is_exhausted(self):
        return self._exhausted

    def uses_disk_storage(self):
        return self._snapshots_on_disk > 0
