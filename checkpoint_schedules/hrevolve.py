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
        _description_, by default (0.0, 0.1)
    rvect : tuple, optional
        _description_, by default (0.0, 0.1)
    cfwd : float, optional
        _description_, by default 1.0
    cbwd : float, optional
        _description_, by default 2.0
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

    # def get_forward_schedule(self):
    #     """Return the hevolve schedule of the forward mode.

    #     Returns
    #     -------
    #     list
    #         Forward schedule list.
    #     """
    #     index_0 = 0
    #     index_1 = None
    #     i = 0
    #     while index_1 is None:
    #         if self._schedule[i].index[1] == self._max_n:
    #             index_1 = i+1
    #             fwd_end_step = self._schedule[i].index[1]
    #         i+=1
    #     self.end_forward = (True, index_1)
    #     self._n = fwd_end_step
    #     return self._schedule[index_0:index_1]
        

    # def get_reverse_schedule(self):
    #     """Return the hevolve schedule of the backward mode.

    #     Returns
    #     -------
    #     list
    #         Reverse schedule list.
    #     """
    #     index_0 = self.end_forward[1]
        
    #     return self._schedule[index_0: len(self._schedule)]

    def iter(self):
        """Iterator.
        """
        def action(i):
            """Provide the actions.

            Parameterss
            ----------
            i : int
                _description_

            Returns
            -------
                _description_

            Raises
            ------
            RuntimeError
                _description_
            RuntimeError
                _description_
            """
            assert i >= 0 and i < len(self._schedule)
            action = self._schedule[i]
            cp_action = action.type
            if cp_action == "Forward":
                n_0, n_1 = action.index
                storage = None
                if n_1 <= n_0:
                    raise RuntimeError("Invalid schedule")
            elif cp_action == "Backward":
                n_0 = action.index
                n_1 = None
                storage = None
            elif cp_action in ["Read", "Write", "Discard"]:
                storage, n_0 = action.index
                n_1 = None
                storage = {0: "RAM", 1: "disk"}[storage]
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")
            return cp_action, (n_0, n_1, storage)

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")
    
        snapshots = set()
        deferred_cp = None

        def write_deferred_cp():
            nonlocal deferred_cp
            if deferred_cp is not None:
                snapshots.add(deferred_cp[0])
                yield Write(*deferred_cp)
                deferred_cp = None

        for i in range(len(self._schedule)-1):
            cp_action, (n_0, n_1, storage) = action(i)

            if cp_action == "Forward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")

                yield Clear(True, True)
                yield Configure(n_0 not in snapshots, False)
                self._n = n_1
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
                self._r += 1
                yield Reverse(n_0, n_0-1)
            elif cp_action == "Read":
                if deferred_cp is not None:
                    raise RuntimeError("Invalid checkpointing state")
                if n_0 == self._max_n - self._r:
                    cp_delete = True
                elif i < len(self._schedule) - 2:
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
            elif cp_action == "Discard":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                r_cp_action, (r_n_0, _, r_storage) = action(i - 2)
                if r_cp_action != "Read" \
                        or r_n_0 != n_0 \
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
