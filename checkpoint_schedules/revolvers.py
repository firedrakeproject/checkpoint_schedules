#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.

from enum import Enum
from .schedule import CheckpointSchedule, Forward, Reverse, Transfer,\
    EndForward, EndReverse
from .revolve_sequences import hrevolve, disk_revolve, periodic_disk_revolve
import logging


__all__ = \
    [
        "RevolveCheckpointSchedule", "RevolverMethod", "StorageLocation"
    ]

class RevolverMethod(Enum):
    """List of checkpointing revolvers that are available.
    """
    HREVOLVE = "hrevolve"
    DISKREVOLVE = "disk_revolve"
    PERIODICDISKREVOLVE = "periodic_disk_revolve"

class StorageLocation(Enum):
    """List of storage level.
    """
    RAM = 0
    DISK = 1
    CHECKPOINT = 2
    NONE = None

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
    f_cost : float, optional
        The cost of advancing the forward one step.
    b_cost : float, optional
        The cost of advancing the adjoint over that step.
    wvect : float, optional
        The write cost associated with saving a forward restart checkpoint to disk.
    rvect : float, optional
        The read cost associated with loading a forward restart checkpoint from disk.
    
    Notes
    -----
    The write and read cost with saving a forward restart checkpoint to RAM is 0.
    """

    def __init__(self, max_n, snap_in_ram, snap_on_disk=0,
                 wvec=(0, 0.1), rvec=(0, 0.1), schedule='hrevolve'):
        super().__init__(max_n)
        self._exhausted = False
        self._snapshots_on_disk = snap_on_disk
        if schedule == RevolverMethod.HREVOLVE.value:
            cvec = (snap_in_ram, snap_on_disk)
            sequence = hrevolve(max_n - 1 , cvec, wvec, rvec)
        elif schedule == RevolverMethod.DISKREVOLVE.value:
            sequence = disk_revolve(max_n - 1, snap_in_ram, wvec[1], rvec[1])
        elif schedule == RevolverMethod.PERIODICDISKREVOLVE.value:
            sequence = periodic_disk_revolve(max_n - 1, snap_in_ram, wd=wvec[1], rd=rvec[1])
        else:
            raise ValueError
        self._schedule = list(sequence)

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
        w_storage = None
        write_ics = False
        write_data = False

        i = 0
        while i < len(self._schedule):
            cp_action, (n_0, n_1, storage) = _convert_action(self._schedule[i])
            if cp_action == "Forward":
                if n_0 != self._n:
                    raise InvalidForwardStep
                self._n = n_1
                w_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i - 1])
                r_cp_action, (r_n0, _, r_storage) = _convert_action(self._schedule[i - 2])
                if (w_cp_action == "Write"
                    or w_cp_action == "Write_disk" 
                    or w_cp_action == "Write_memory"):
                    if w_n0 != n_0:
                        raise InvalidActionIndex
                    write_ics = True
                    write_data = False
                    snapshots.add(w_n0)
                elif (w_cp_action == "Write_Forward"
                    or w_cp_action == "Write_Forward_memory"):
                    if w_n0 != n_1:
                        raise InvalidActionIndex
                    write_ics = False
                    write_data = True
                else:
                    write_ics = False
                    write_data = False
                    w_storage = StorageLocation(None).name
                yield Forward(n_0, n_1, write_ics, write_data, w_storage)
                if self._n == self._max_n:
                    if self._r != 0:
                        raise InvalidReverseStep
                    yield EndForward()
            elif cp_action == "Backward":
                if n_0 != self._n:
                    raise InvalidActionIndex
                if n_0 != self._max_n - self._r:
                    raise InvalidForwardStep
                self._r += 1
                yield Reverse(n_0, n_1, clear_fwd_data=True)
            elif cp_action == "Read":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])
                if n_cp_action == "Write":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, w_storage)
                elif n_cp_action == "Write_Forward":
                    if f_cp_action != "Forward":
                        raise InvalidRevolverAction
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, StorageLocation(2).name)
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, StorageLocation(2).name)
                else:
                    raise InvalidRevolverAction
            elif cp_action == "Read_disk":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])
                if n_cp_action == "Write_memory":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, w_storage)
                elif n_cp_action == "Write_Forward_memory":
                    if f_cp_action != "Forward":
                        raise InvalidRevolverAction
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, StorageLocation(2).name)
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, StorageLocation(2).name)
                else:
                    raise InvalidRevolverAction
            elif cp_action == "Read_memory":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])

                if n_cp_action == "Write_Forward_memory":
                    if f_cp_action != "Forward":
                        raise InvalidRevolverAction
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, StorageLocation(2).name)
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, StorageLocation(2).name)
                else:
                    raise InvalidRevolverAction
            elif (cp_action == "Write" or cp_action == "Write_disk"
                  or cp_action == "Write_memory"):
                if n_0 != self._n:
                    raise InvalidActionIndex
            elif cp_action == "Write_Forward":
                if n_0 != self._n + 1:
                    raise InvalidActionIndex
                d_cp_action, (d_n0, _, w_storage) = _convert_action(self._schedule[i + 3])
                if (d_cp_action != "Discard_Forward"
                    or d_n0 != n_0 or w_storage != storage):
                    if w_n0 != n_0:
                        raise InvalidActionIndex
                    write_ics = True
                    write_data = False
            elif cp_action == "Write_Forward_memory":
                if n_0 != self._n + 1:
                    raise InvalidActionIndex
                d_cp_action, (d_n0, _, w_storage) = _convert_action(self._schedule[i + 3])
                if (d_cp_action != "Discard_Forward_memory"
                    or d_n0 != n_0 or w_storage != storage):
                    if w_n0 != n_0:
                        raise InvalidActionIndex
            elif cp_action == "Discard" or cp_action == "Discard_memory":
                if i < 2:
                    raise InvalidRevolverAction
                snapshots.remove(n_0)
                yield Transfer(n_0, storage, StorageLocation(None).name, delete=True)
            elif cp_action == "Discard_Forward" or cp_action == "Discard_Forward_memory":
                if n_0 != self._n:
                    raise InvalidActionIndex
            else:
                raise InvalidRevolverAction
            i += 1
        # if len(snapshots) > 0:
        #     raise RuntimeError("Unexpected snapshot number.")
        
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
            raise RuntimeError("Invalid forward indexes.")
        storage = None
    elif cp_action == "Backward":
        n_0, n_1 = action_n.index
        if n_0 <= n_1:
            raise RuntimeError("Invalid backward indexes.")
        storage = None
    elif cp_action in ["Read", "Write", "Discard",
                       "Write_Forward", "Discard_Forward"]:
        storage, n_0 = action_n.index
        n_1 = None
        storage = {0: StorageLocation(0).name, 1: StorageLocation(1).name}[storage]
    elif cp_action in ["Write_Forward_memory",
                       "Discard_Forward_memory"]:
        n_0 = action_n.index
        n_1 = None
        storage = {0: StorageLocation(0).name}[0]
    elif cp_action in ["Read_disk", "Write_disk", "Discard_disk"]:
        n_0 = action_n.index
        n_1 = None
        storage = 1
        storage = {1: StorageLocation(1).name}[storage]
    elif cp_action in ["Read_memory", "Write_memory", "Discard_memory"]:
        n_0 = action_n.index
        n_1 = None
        storage = 0
        storage = {0: StorageLocation(0).name}[storage]
    else:
        raise InvalidRevolverAction
    return cp_action, (n_0, n_1, storage)


class InvalidForwardStep(IndexError):
    "The forward step is not correct."
    pass


class InvalidReverseStep(IndexError):
    "The reverse step is not correct."
    pass

class InvalidRevolverAction(Exception):
    "The action is not expected for this iterator."
    pass


class InvalidActionIndex(IndexError):
    "The index of the action is not correct."
    pass