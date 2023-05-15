#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add corect license text
from .schedule import CheckpointSchedule, Clear, Configure, Forward, Reverse, \
    Read, Write, EndForward, EndReverse, WriteForward
from .revolve_sequences import hrevolve, disk_revolve
import logging

__all__ = \
    [
        "RevolveCheckpointSchedule"
    ]


class RevolveCheckpointSchedule(CheckpointSchedule):
    """H-Revolve checkpointing schedule.

    Attributes
    ----------
    max_n : int
        Total checkpoint of a foward solver.
    snapshots_in_ram : int
        Number of checkpoints save in RAM.
    snapshots_on_disk : int
        Number of checkpoints save in disk.
    wvect : tuple
        Cost of writing to each level of memory.
    rvect : tuple
        Cost of reading from each level of memory.
    cbwd : float
        Cost of the backward steps.
    cfwd : float
        Cost of the forward steps.
    """
    def __init__(self, max_n, snapshots_in_ram=None, snapshots_on_disk=None, *,
                 wvect=(0.0, 0.1), rvect=(0.0, 0.1), cfwd=1.0, cbwd=1.0,
                 revolve_sequence="hrevolve", **kwargs):
        
        super().__init__(max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._exhausted = False
        self.end_forward = (False, None)
        cvect = (snapshots_in_ram, snapshots_on_disk)
        if revolve_sequence == "hrevolve":
            assert snapshots_in_ram is not None
            assert snapshots_on_disk is not None
            schedule = hrevolve(max_n, cvect, wvect, rvect,
                                cfwd=cfwd, cbwd=cbwd, **kwargs)
        elif revolve_sequence == "disk_revolve":
            schedule = disk_revolve(l=max_n, cm=snapshots_on_disk,
                                    wd=wvect[1], rd=rvect[1], ub=cbwd)
          
        self._schedule = list(schedule)
        self.snapshots = set()
        self.forward_data = set()

    def iter(self):
        """Iterator.
        """
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
            elif cp_action in ["Read", "Read_disk", "Read_memory",
                               "Write",  "Write_disk", "Write_memory",
                               "Write_Forward", "Write_Forward_memory",
                               "Write_Forward_disk", "Discard",
                               "Discard_disk", "Discard_memory",
                               "Discard_Forward", "Discard_Forward_disk",
                               "Discard_Forward_memory"]:
                if (cp_action == "Write_disk" or "Write_Forward_disk"
                    or "Read_disk" or "Discard_disk"
                    or "Discard_Forward_disk"):
                    n_0 = action.index
                    storage = {0: "RAM", 1: "disk"}[1]
                if (cp_action == "Write_memory" or "Write_Forward_memory"
                    or "Read_memory" or "Discard_memory"
                    or "Discard_Forward_memory"):
                    n_0 = action.index
                    storage = {0: "RAM", 1: "disk"}[0]
                else:
                    storage, n_0 = action.index
                    storage = {0: "RAM", 1: "disk"}[storage]
                n_1 = None
                
            else:
                print(cp_action)
                raise RuntimeError(f"Unexpected action: {cp_action:s}")
            return cp_action, (n_0, n_1, storage)

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")
        
        deferred_cp = None

        def write_deferred_cp():
            nonlocal deferred_cp
            if deferred_cp is not None:
                self.snapshots.add(deferred_cp[0])
                yield Write(*deferred_cp)
                deferred_cp = None

        for i in range(len(self._schedule)):
            cp_action, (n_0, n_1, storage) = action(i)

            if cp_action == "Forward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")            
                
                yield Clear(True, True)
                w_cp_action, _ = action(i + 1)
                if (w_cp_action == "Write_Forward"
                    or  w_cp_action == "Write_Forward_disk"
                    or  w_cp_action == "Write_Forward_memory"):
                    yield Configure(False, True)
                if (w_cp_action == "Write"
                    or  w_cp_action == "Write_disk"
                    or  w_cp_action == "Write_memory"):
                    yield Configure(True, False)

                self._n = n_1
                yield Forward(n_0, n_1)
                if self._n == self._max_n:
                    if self._r != 0:
                        raise RuntimeError("Invalid checkpointing state")
                    yield EndForward(True)
            elif cp_action == "Backward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                if n_0 != self._max_n - self._r:
                    raise RuntimeError("Invalid checkpointing state")
                
                yield from write_deferred_cp()
                if i < len(self._schedule) - 1:
                    d_cp_action, (d_n_0, _, d_storage) = action(i + 1)
                    if (d_cp_action == "Discard_Forward" 
                        or d_cp_action == "Discard_Forward_disk"
                        or d_cp_action == "Discard_Forward_memory"):
                        self.forward_data.remove(n_0)

                if n_0 > 0:
                    self._r += 1
                    yield Reverse(n_0, n_0-1)
                else:
                    Reverse(n_0, n_0)
            elif cp_action == "Read" or cp_action == "Read_disk" or cp_action == "Read_memory":
                if deferred_cp is not None:
                    raise RuntimeError("Invalid checkpointing state")

                if n_0 == self._max_n - self._r:
                    cp_delete = True
                elif i < len(self._schedule) - 2:
                    d_cp_action, (d_n_0, _, d_storage) = action(i + 2)
                    if (d_cp_action == "Discard"
                        or d_cp_action == "Discard_disk"
                        or d_cp_action == "Discard_memory"):
                        if d_n_0 != n_0 or d_storage != storage:
                            raise RuntimeError("Invalid schedule")
                        cp_delete = True
                    else:
                        cp_delete = False
                yield Clear(True, True) 
                if cp_delete:
                    self.snapshots.remove(n_0)
                self._n = n_0
                yield Read(n_0, storage, cp_delete)
            elif (cp_action == "Write"
                  or cp_action == "Write_disk"
                  or cp_action == "Write_memory"):
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                deferred_cp = (n_0, storage)
                yield from write_deferred_cp()
                
                if i > 0:
                    r_cp_action, (r_n_0, _, _) = action(i - 1)
                    if ((cp_action == "Write" and r_cp_action == "Read") 
                        or (r_cp_action == "Write_disk" and r_cp_action == "Read_disk")
                        or (r_cp_action == "Write_memory" and r_cp_action == "Read_memory")):
                        if r_n_0 != n_0:
                            raise RuntimeError("Invalid schedule")
                        yield from write_deferred_cp()
            elif (cp_action == "Write_Forward"
                  or cp_action == "Write_Forward_memory"
                  or cp_action == "Write_Forward_disk"):
                d_cp_action, (d_n_0, _, d_storage) = action(i + 2)
                if ((cp_action == "Write_Forward" and d_cp_action != "Discard_Forward") 
                    or (cp_action == "Write_Forward_memory" and d_cp_action != "Discard_Forward_memory")
                    or (cp_action == "Write_Forward_disk" and d_cp_action != "Discard_Forward_disk")):

                    raise RuntimeError("Invalid checkpointing state")

                self.forward_data.add(n_0)
                yield WriteForward(n_0, storage)

            elif (cp_action == "Discard" or "Discard_disk" or "Discard_memory"):
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                r_cp_action, (r_n_0, _, r_storage) = action(i - 2)
                if ((cp_action == "Discard" and r_cp_action != "Read")
                    or (cp_action == "Discard_disk" and r_cp_action != "Read_disk")
                    or (cp_action == "Discard_memory" and r_cp_action != "Read_memory")
                    or r_n_0 != n_0 or r_storage != storage):
                    raise RuntimeError("Invalid schedule")
            elif (cp_action == "Discard_Forward"
                  or cp_action == "Discard_Forward_disk"
                  or cp_action == "Discard_Forward_memory"):
                if i < 2:
                    raise RuntimeError("Invalid schedule")
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")

        if len(self.snapshots) != 0:
            raise RuntimeError("Invalid checkpointing state")

        yield Clear(True, True)

        self._exhausted = True
        yield EndReverse(True)

    def is_exhausted(self):
        return self._exhausted

    def uses_disk_storage(self):
        return self._snapshots_on_disk > 0
