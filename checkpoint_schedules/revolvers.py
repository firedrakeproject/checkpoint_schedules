#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from .schedule import CheckpointSchedule, Forward, Reverse, Copy,\
    EndForward, EndReverse, StorageLocation
from .revolve_sequences import hrevolve

__all__ = \
    [
        "RevolveCheckpointSchedule"
    ]


class RevolveCheckpointSchedule(CheckpointSchedule):
    """This object allows to iterate over a sequence
    of the checkpoint schedule actions. 
 
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
    wvect : tuple, optional
        The write cost associated with saving a forward restart checkpoint to
        RAM and disk.
    rvect : tuple, optional
        The read cost associated with loading a forward restart checkpoint
        from RAM and disk.
    self._schedule : list
        H-Revolve sequence of operations.
    
    """

    def __init__(self, max_n, snap_in_ram, snap_on_disk, fwd_cost=1.0,
                 bwd_cost=1.0, wvec=(0, 0.5), rvec=(0, 0.5)):
        super().__init__(max_n)
        self._exhausted = False
        self._snapshots_on_disk = snap_on_disk
        cvec = (snap_in_ram, snap_on_disk)
        sequence = hrevolve(max_n - 1, cvec, wvec, rvec, fwd_cost, bwd_cost)
        self._schedule = list(sequence)

    def iter(self):
        """A checkpoint schedules iterator.

        Yields
        ------
        action, (n0, n1, storage)
            Schedule actions, step `n0`, step `n1` and the storage type.

        Notes
        -----
        The actions and the data `n0`, `n1` and `sorage` follow the H-Revolve 
        schedule.
        
        """
        if self._max_n is None:
            raise RuntimeError("Invalid forward steps number.")

        snapshots = set()
        w_storage = None
        write_ics = False
        adj_deps = False

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
                    adj_deps = False
                    snapshots.add(w_n0)
                elif (w_cp_action == "Write_Forward"
                    or w_cp_action == "Write_Forward_memory"):
                    if w_n0 != n_1:
                        raise InvalidActionIndex
                    write_ics = False
                    adj_deps = True
                else:
                    write_ics = False
                    adj_deps = False
                    w_storage = StorageLocation(None).name
                yield Forward(n_0, n_1, write_ics, adj_deps, w_storage)
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
                yield Reverse(n_0, n_1, clear_adj_deps=True)
            elif cp_action == "Read":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])
                if n_0 == self._max_n - self._r - 1:
                    delete = True
                else:
                    delete = False
                yield Copy(n_0, storage, StorageLocation(-1).name, delete=delete)
            elif cp_action == "Read_disk":
                self._n = n_0
                n_cp_action, (w_n0, _, w_storage) = _convert_action(self._schedule[i + 1])
                f_cp_action, (f_n0, _, _) = _convert_action(self._schedule[i + 2])
                if n_cp_action == "Write_memory":
                    assert n_0 == w_n0
                    yield Copy(n_0, storage, w_storage)
                elif n_cp_action == "Write_Forward_memory":
                    if f_cp_action != "Forward":
                        raise InvalidRevolverAction
                    assert n_0 == f_n0
                    yield Copy(n_0, storage, StorageLocation(-1).name)
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Copy(n_0, storage, StorageLocation(-1).name)
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
                    yield Copy(n_0, storage, StorageLocation(-1).name)
                elif n_cp_action == "Forward":
                    assert n_0 == w_n0
                    yield Copy(n_0, storage, StorageLocation(-1).name)
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
                    adj_deps = False
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
                # yield Copy(n_0, storage, StorageLocation(None).name, delete=True)
            elif cp_action == "Discard_Forward" or cp_action == "Discard_Forward_memory":
                if n_0 != self._n:
                    raise InvalidActionIndex
            else:
                raise InvalidRevolverAction
            i += 1
        if len(snapshots) > self._snapshots_on_disk:
            raise RuntimeError("Unexpected snapshot number.")
        
        self._exhausted = True
        yield EndReverse(True)


    def is_exhausted(self):
        """Indicate whether the schedule has concluded.

        Returns
        -------
        bool
            End the reverse computation if ``True``.
        """
        return self._exhausted
    
    def uses_disk_storage(self):
        """Indicate whether the `DISK` storage level is used.

        Returns
        -------
        bool
            `DISK` storage is used to store the checkpoint data if ``True``.
        """
        return self._snapshots_on_disk > 0


def _convert_action(action):
    """Read an H-Revolve operation attributes required to 
    define the next checkpoint action.

    Parameters
    ----------
    action_n : h_revolve.operation
        An operation from the H-revolve sequence.
    
    Notes
    -----
    The H-Revolve operation has type and index attributes used to interpret the next action.
    For instance, if operation type is `Forward`, the operation index that is a tuple `(n0, n1)`, 
    which it read as the next action aims to execute the forward solver from the step n0 to step n1.
    Write is another H-Revolve operation type 

    Returns
    -------
    str, tuple(int, int , str)
        Return the operation name, steps `n_0`, step `n_1` and the storage level (either RAM or disk).

    """
    cp_action = action.type
    if cp_action == "Forward":
        n_0, n_1 = action.index
        if n_1 <= n_0:
            raise RuntimeError("Invalid forward indexes.")
        storage = None
    elif cp_action == "Backward":
        n_0, n_1 = action.index
        if n_0 <= n_1:
            raise RuntimeError("Invalid backward indexes.")
        storage = None
    elif cp_action in ["Read", "Write", "Discard",
                       "Write_Forward", "Discard_Forward"]:
        storage, n_0 = action.index
        n_1 = None
        storage = {0: StorageLocation(0).name, 1: StorageLocation(1).name}[storage]
    elif cp_action in ["Write_Forward_memory",
                       "Discard_Forward_memory"]:
        n_0 = action.index
        n_1 = None
        storage = {0: StorageLocation(0).name}[0]
    elif cp_action in ["Read_disk", "Write_disk", "Discard_disk"]:
        n_0 = action.index
        n_1 = None
        storage = 1
        storage = {1: StorageLocation(1).name}[storage]
    elif cp_action in ["Read_memory", "Write_memory", "Discard_memory"]:
        n_0 = action.index
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