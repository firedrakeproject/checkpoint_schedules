#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add corect license text
from .hrevolve_sequence import hrevolve
import logging

__all__ = \
    [
        "HRevolveCheckpointSchedule"
    ]


class HRevolveCheckpointSchedule():
    """H-Revolve checkpointing schedule.

    Attributes
    ----------
    max_n : int
        Total checkpoint of a foward solver.
    snapshots_in_ram : int
        Number of checkpoints saves in RAM.
    snapshots_on_disk : int
        Number of checkpoints saves in disk.
    wvect : tuple, optional
        Cost of writing to each level of memory.
    rvect : tuple, optional
        Cost of reading from each level of memory.
    cfwd : float, optional
        Cost of the forward steps.
    cfwd : float, optional
        Cost of the backward steps.
    """
    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 wvect=(0.0, 0.1), rvect=(0.0, 0.1), cfwd=1.0, cbwd=2.0, **kwargs):
        
        # super().__init__(max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._exhausted = False
        self._max_n = max_n
        self.end_forward = (False, None)
        cvect = (snapshots_in_ram, snapshots_on_disk)
        schedule = hrevolve(max_n, cvect, wvect, rvect,
                            cfwd=cfwd, cbwd=cbwd, **kwargs)
        
        self._schedule = list(schedule)

    def get_forward_schedule(self):
        """Return the hevolve schedule of the forward mode.

        Returns
        -------
        list
            Forward schedule list.
        """
        index_0 = 0
        index_1 = None
        i = 0
        while index_1 is None:
            if self._schedule[i].index[1] == self._max_n:
                index_1 = i+1
            i+=1
        self.end_forward = (True, index_1)
        return self._schedule[index_0:index_1]
        

    def get_reverse_schedule(self):
        """Return the hevolve schedule of the backward mode.

        Returns
        -------
        list
            Reverse schedule list.
        """
        index_0 = self.end_forward[1]
        return self._schedule[index_0: len(self._schedule)]

  