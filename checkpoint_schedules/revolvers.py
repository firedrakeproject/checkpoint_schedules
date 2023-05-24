#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add corect license text
from .schedule import CheckpointSchedule, Clear, Configure, Forward, Reverse, \
    Read, Write, EndForward, EndReverse, WriteForward, Delete
from .revolve_sequences import hrevolve
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
        The number of forward steps in the initial forward calculation.
    snapshots_in_ram : int
        The maximum number of forward restart checkpoints to store in memory.
    snapshots_on_disk : int
        The maximum number of forward restart checkpoints to store on disk.
    wvect : tuple
        A two element defining the write cost associated with saving a forward 
        restart checkpoint to RAM (first element) and disk (second element).
    rvect : tuple
        A two element defining the read cost associated with loading a forward 
        restart checkpoint from RAM (first element) and disk (second element).
    cfwd : float
        The cost of advancing the forward one step.
    cbwd : float
        The cost of advancing the forward one step, storing non-linear dependency
        data, and then advancing the adjoint over that step.
    """
    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 wvect=(0.0, 0.1), rvect=(0.0, 0.1), cfwd=1.0, cbwd=1.0, **kwargs):
        
        super().__init__(max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._exhausted = False
        self.end_forward = (False, None)
        cvect = (snapshots_in_ram, snapshots_on_disk)
        schedule = hrevolve(max_n, cvect, wvect, rvect,
                            cfwd=cfwd, cbwd=cbwd, **kwargs)
        
        self._schedule = list(schedule)
        self._r = self._n = 0
        self.snapshots = set()
        self.forward_data = set()

    def iter(self):
        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")
        
        deferred_cp = None

        def write_deferred_cp():
            nonlocal deferred_cp
            if deferred_cp is not None:
                # yield Clear(True, True)
                yield Configure(True, False)
                self.snapshots.add(deferred_cp[0])
                yield Write(*deferred_cp)
                deferred_cp = None
        i = 0
        while i < len(self._schedule):
            cp_action, (n_0, n_1, storage) = action_info(self._schedule[i])

            if cp_action == "Forward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")            
                self._n = n_1
                yield Forward(n_0, n_1)
            elif cp_action == "Backward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                # if n_0 != self._max_n - self._r:
                #     raise RuntimeError("Invalid checkpointing state")

                yield from write_deferred_cp()
                self._n = n_0
                self._r += 1
                yield Reverse(n_0, n_1)
                df_cp_action, (f_n_0, _, f_storage) = action_info(self._schedule[i + 1])
                assert df_cp_action == "Discard_Forward"
                assert f_n_0 == n_0
                yield Delete(n_0, f_storage, delete_data=True)
                if i < len(self._schedule) - 2:
                    dic_cp_action, (ic_n_0, _, dic_storage) = action_info(self._schedule[i + 2])
                    if dic_cp_action == "Discard":
                        if ic_n_0 != n_1:
                            raise RuntimeError("Invalid schedule")
                        yield Delete(n_1, dic_storage, delete_ics=True)
                  
                        self.snapshots.remove(n_1)
            elif cp_action == "Read":
                if deferred_cp is not None:
                    raise RuntimeError("Invalid checkpointing state")
                self._n = n_0
                yield Read(n_0, storage)
            elif cp_action == "Write":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                deferred_cp = (n_0, storage)
                yield from write_deferred_cp()

                

                # if i > 0:
                #     r_cp_action, (r_n_0, _, _) = action_info(self._schedule[i - 1])
                #     if r_cp_action == "Read":
                #         if r_n_0 != n_0:
                #             raise RuntimeError("Invalid schedule")
                #         yield from write_deferred_cp()

            elif cp_action == "Write_Forward":
                d_cp_action, (d_n_0, _, d_storage) = action_info(self._schedule[i + 2])
                if d_cp_action != "Discard_Forward":
                    raise RuntimeError("Invalid checkpointing state")
                
                if d_n_0 != n_0 or d_storage != storage:
                    raise RuntimeError("Invalid schedule")
                
                self.forward_data.add(n_0)
                # yield Clear(True, True)
                yield Configure(False, True)
                yield WriteForward(n_0, storage)
                if self._n == self._max_n:
                    if self._r != 0:
                        raise RuntimeError("Invalid checkpointing state")
                    yield EndForward()

            elif cp_action == "Discard":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                # r_cp_action, (r_n_0, _, r_storage) = action_info(self._schedule[i - 2])
                # if r_cp_action != "Read" \
                #         or r_n_0 != n_0 \
                #         or r_storage != storage:
                #     raise RuntimeError("Invalid schedule")
            elif cp_action == "Discard_Forward":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                r_cp_action, (df_n_0, _, df_storage) = action_info(self._schedule[i - 2])
                if r_cp_action != "Write_Forward" \
                        or df_n_0 != n_0 \
                        or df_storage != storage:
                    raise RuntimeError("Invalid schedule")
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")
            i += 1
        # yield Delete(True, True)
        # if len(self.snapshots) != 0:
        #     raise RuntimeError("Invalid checkpointing state")

        # yield Clear(True, True)
        
        self._exhausted = True
        yield EndReverse(True)

    def is_exhausted(self):
        return self._exhausted

    def uses_disk_storage(self):
        return self._snapshots_on_disk > 0


def action_info(action_n):
    """Return the action at step "n" and its informations.

    Parameters
    ----------
    action_n : _type_
        Action at the time "n".

    Returns
    -------
    _type_
        _description_
    """
    cp_action = action_n.type
    if cp_action == "Forward":
        n_0, n_1 = action_n.index
        if n_1 <= n_0:
            raise RuntimeError("Invalid action_n")
        storage = None
    elif cp_action == "Backward":
        n_0, n_1 = action_n.index
        storage = None
    elif cp_action in ["Read", "Write",
                        "Write_Forward", "Discard",
                        "Discard_Forward"]:
        storage, n_0 = action_n.index
        n_1 = None
        storage = {0: "RAM", 1: "disk"}[storage]
    else:
        raise RuntimeError(f"Unexpected action: {cp_action:s}")
    return cp_action, (n_0, n_1, storage)
