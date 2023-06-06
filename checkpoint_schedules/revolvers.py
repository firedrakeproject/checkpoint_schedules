#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.

from .schedule import CheckpointSchedule, Forward, Reverse, Transfer,\
    EndForward, EndReverse
from .revolve_sequences import hrevolve, disk_revolve, periodic_disk_revolve
import logging
from enum import Enum

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

    def __init__(self, max_n, snap_in_ram, snaps_on_disk=0, f_cost=1, b_cost=1,
                 w_cost=0.1, r_cost=0.1, schedule=0):
        super().__init__(max_n)
        self._exhausted = False
        self._snapshots_on_disk = snaps_on_disk
        schedule = RevolverMethod(schedule).sequence(max_n, snap_in_ram,
                                                     snaps_on_disk, f_cost,
                                                     b_cost, r_cost, w_cost)
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
                    w_storage = None
                yield Forward(n_0, n_1, write_ics, write_data, w_storage)
                if self._n == self._max_n:
                    if self._r != 0:
                        raise InvalidBackwardStep
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
                        raise InvalidRevolverSchedule
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, "checkpoint")
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, "checkpoint")
                else:
                    raise InvalidRevolverSchedule
            elif cp_action == "Read_disk":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])
                if n_cp_action == "Write_memory":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, w_storage)
                elif n_cp_action == "Write_Forward_memory":
                    if f_cp_action != "Forward":
                        raise InvalidRevolverSchedule
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, "checkpoint")
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, "checkpoint")
                else:
                    raise InvalidRevolverSchedule
            elif cp_action == "Read_memory":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])

                if n_cp_action == "Write_Forward_memory":
                    if f_cp_action != "Forward":
                        raise InvalidRevolverSchedule
                    assert n_0 == f_n0
                    yield Transfer(n_0, storage, "checkpoint")
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Transfer(n_0, storage, "checkpoint")
                else:
                    raise InvalidRevolverSchedule
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
                    raise InvalidRevolverSchedule
                snapshots.remove(n_0)
                yield Transfer(n_0, storage, None, delete=True)
            elif cp_action == "Discard_Forward" or cp_action == "Discard_Forward_memory":
                if n_0 != self._n:
                    raise InvalidActionIndex
            else:
                raise InvalidRevolverAction
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
                       "Write_Forward", "Discard_Forward"]:
        storage, n_0 = action_n.index
        n_1 = None
        storage = {0: "RAM", 1: "disk"}[storage]
    elif cp_action in ["Write_Forward_memory",
                       "Discard_Forward_memory"]:
        n_0 = action_n.index
        n_1 = None
        storage = {0: "RAM"}[0]
    elif cp_action in ["Read_disk", "Write_disk", "Discard_disk"]:
        n_0 = action_n.index
        n_1 = None
        storage = 1
        storage = {1: "disk"}[storage]
    elif cp_action in ["Read_memory", "Write_memory", "Discard_memory"]:
        n_0 = action_n.index
        n_1 = None
        storage = 0
        storage = {0: "RAM"}[storage]
    else:
        raise RuntimeError(f"Unexpected action: {cp_action:s}")
    return cp_action, (n_0, n_1, storage)


class RevolverMethod(Enum):
    """Set the revolver method.
    """
    HREVOLVE = 0
    DISKREVOLVE = 1
    PERIODICDISKREVOLVE = 2

    def sequence(self, max_n, snap_in_ram, snap_disk, f_cost, b_cost, r_cost, w_cost):
        """Determine the revolve sequence

        Parameters
        ----------
        max_n : _type_
            _description_
        snap_in_ram : _type_
            _description_
        snap_disk : _type_
            _description_
        f_cost : _type_
            _description_
        b_cost : _type_
            _description_
        r_cost : _type_
            _description_
        w_cost : _type_
            _description_

        Returns
        -------
        object
            Revolver sequence.
        """
        if self.name == 'DISKREVOLVE':
            assert snap_in_ram >= 0
            params = {
                        "uf": f_cost,            # Cost of a forward step.
                        "ub": b_cost,            # Cost of a backward step.
                        "up": 1,                 # Cost of the loss function.
                        "wd": w_cost,            # Cost of writing to disk.
                        "rd": r_cost,            # Cost of reading from disk.
                        "mx": None,              # Size of the period (defaults to the optimal).
                        "one_read_disk": False,  # Disk checkpoints are only read once.
                        "fast": False,           # Use the clode formula for mx.
                        "concat": 0,             # Level of sequence concatenation.
                        }
            return disk_revolve(max_n - 1, snap_in_ram, **params)
        
        elif self.name == 'HREVOLVE':
            assert snap_in_ram > 0
            wvec = [0, w_cost]           
            rvec = [0, r_cost]
            params = {
                        "uf": f_cost,            # Cost of a forward step.
                        "ub": b_cost,            # Cost of a backward step.
                        "up": 1,                 # Cost of the loss function.
                        "rd" : rvec,
                        "wd" : wvec,
                        "mx": None,              # Size of the period (defaults to the optimal).
                        "one_read_disk": False,  # Disk checkpoints are only read once.
                        "fast": False,           # Use the clode formula for mx.
                        "concat": 0,             # Level of sequence concatenation.
                        }
            cvec = [snap_in_ram, snap_disk]
            
            return hrevolve(max_n - 1 , cvec, wvec, rvec, **params)
        elif self.name == 'PERIODICDISKREVOLVE':
            assert snap_in_ram > 0
            params = {
                        "uf": f_cost,            # Cost of a forward step.
                        "ub": b_cost,            # Cost of a backward step.
                        "wd": w_cost,            # Cost of writing to disk.
                        "rd": r_cost,            # Cost of reading from disk.
                        "up": 1,                 # Cost of the loss function.
                        "mx": None,              # Size of the period (defaults to the optimal).
                        "one_read_disk": False,  # Disk checkpoints are only read once.
                        "fast": False,           # Use the clode formula for mx.
                        "concat": 0,             # Level of sequence concatenation.
                        }
            return periodic_disk_revolve(max_n - 1, snap_in_ram, **params)


class InvalidForwardStep(Exception):
    "Raised when the forward step is invalid."
    pass


class InvalidBackwardStep(Exception):
    "Raised when the forward step is invalid."
    pass


class InvalidWriteindex(Exception):
    "Raised when the forward step is invalid."
    pass


class InvalidReadindex(Exception):
    "Raised when the forward step is invalid."
    pass


class InvalidRevolverSchedule(Exception):
    "Raised when the forward step is invalid."
    pass


class InvalidRevolverAction(Exception):
    "Raised when the forward step is invalid."
    pass


class InvalidActionIndex(Exception):
    "Raised when the forward step is invalid."
    pass