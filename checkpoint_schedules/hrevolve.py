#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""..."""

from .schedule import CheckpointSchedule, Forward, Reverse, Copy, Move, \
    EndForward, EndReverse, StorageType
from .hrevolve_sequences import hrevolve, disk_revolve, periodic_disk_revolve,\
      revolve

__all__ = \
    [
        "HRevolve", "DiskRevolve", "PeriodicDiskRevolve",
        "Revolve"
    ]


class RevolveCheckpointSchedule(CheckpointSchedule):
    """This object allows to iterate over a sequence
    of the checkpoint schedule actions. 
 
    Attributes
    ----------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snap_in_ram : int
        The maximum steps to store the forward checkpoints in RAM.
    snap_on_disk : int
        The maximum steps to store the forward checkpoints on disk.
    schedule : list
        A sequence of operations given by a revolver algorithm.

    Notes
    -----
    This object is written to interpret the revolver algorithmics discussed in [1],
    which means that the revolver algorithmics are used to build the following sequence:
    H-Revolve, Disk Revolve, Periodic Disk Revolve and Revolve.

    [1] Herrmann, J. and Pallez (Aupy), G.. "H-Revolve: a framework
    for adjoint computation on synchronous hierarchical platforms."
    ACM Transactions on Mathematical Software (TOMS) 46.2 (2020): 1-25.
    DOI: https://doi.org/10.1145/3378672.
    """

    def __init__(self, max_n, snap_in_ram, snap_on_disk, schedule):
        super().__init__(max_n)
        self._exhausted = False
        self._snapshots_on_disk = snap_on_disk
        self._snapshots_in_ram = snap_in_ram
        self._schedule = schedule

    def _iterator(self):
        """A checkpoint schedules iterator.

        Yields
        ------
        checkpoint_schedules.schedule.CheckpointScheduleAction
            The next action in the schedule.
        
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
                    w_storage = StorageType.FWD_RESTART
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
            elif (cp_action == "Read"
                  or cp_action == "Read_memory"
                  or cp_action == "Read_disk"):
                self._n = n_0
                
                if n_0 == self._max_n - self._r - 1:
                    snapshots.remove(n_0)
                    yield Move(n_0, storage, StorageType.FWD_RESTART)
                else:
                    yield Copy(n_0, storage, StorageType.FWD_RESTART)     
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
            elif cp_action == "Discard_Forward" or cp_action == "Discard_Forward_memory":
                if n_0 != self._n:
                    raise InvalidActionIndex
            else:
                raise InvalidRevolverAction
            i += 1
        if len(snapshots) > self._snapshots_on_disk + self._snapshots_in_ram:
            raise RuntimeError("Unexpected snapshot number.")
        
        self._exhausted = True
        yield EndReverse()

    @property
    def is_exhausted(self):
        """Indicate whether the schedule has concluded.

        Returns
        -------
        bool
            End the reverse computation if ``True``.
        """
        return self._exhausted
    
    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used in this schedule.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        assert storage_type in StorageType

        if storage_type == StorageType.DISK:
            return self._snapshots_on_disk > 0
        elif storage_type == StorageType.RAM:
            return self._snapshots_in_ram > 0
        


class HRevolve(RevolveCheckpointSchedule):
    """H-Revolve checkpointing schedule.
    
    Atributes
    ---------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snap_in_ram : int
        The maximum steps to store the forward checkpoints in RAM.
    snap_on_disk : int
        The maximum steps to store the forward checkpoints on disk.
    uf : float
        The cost of advancing the forward over one step.
    ub : float
        The cost of advancing the adjoint over one step.
    wd : float
        The cost of writing the checkpoint data in disk.
    rd : float
        The cost of reading the checkpoint data from disk.
    
    Notes
    -----
    The H-Revolve schedule is described in [1].

    [1] Herrmann, J. and Pallez (Aupy), G.. "H-Revolve: a framework
    for adjoint computation on synchronous hierarchical platforms."
    ACM Transactions on Mathematical Software (TOMS) 46.2 (2020): 1-25.
    DOI: https://doi.org/10.1145/3378672.

    """
    def __init__(self, max_n, snap_in_ram, snap_on_disk, uf=1, ub=1, wd=2, rd=2):
        cvec = (snap_in_ram, snap_on_disk)
        wc = [0, wd]
        rc = [0, rd]
        schedule = list(hrevolve(max_n - 1, cvec, wc, rc, uf, ub))
        super().__init__(max_n, snap_in_ram, snap_on_disk, schedule)
        

class DiskRevolve(RevolveCheckpointSchedule):
    """Disk Revolve checkpointing schedule.

    Atributes
    ---------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snap_in_ram : int
        The maximum steps to store the forward checkpoints in RAM.
    uf : float
        The cost of advancing the forward over one step.
    ub : float
        The cost of advancing the adjoint over one step.
    wd : float
        The cost of writing the checkpoint data in disk.
    rd : float
        The cost of reading the checkpoint data from disk.
    
    Notes
    -----
    The H-Revolve schedule is described in [1].
    """

    def __init__(self, max_n, snap_in_ram, uf=1, ub=1, wd=2, rd=2):
        schedule = list(disk_revolve(max_n - 1, snap_in_ram, wd, rd, uf, ub))
        super().__init__(max_n, snap_in_ram, max_n - snap_in_ram, schedule)


class PeriodicDiskRevolve(RevolveCheckpointSchedule):
    """Periodic Disk Revolve checkpointing schedule.
    
    Atributes
    ---------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snap_in_ram : int
        The maximum steps to store the forward checkpoints in RAM.
    uf : float
        The cost of advancing the forward over one step.
    ub : float
        The cost of advancing the adjoint over one step.
    wd : float
        The cost of writing the checkpoint data in disk.
    rd : float
        The cost of reading the checkpoint data from disk.

    Notes
    -----
    The H-Revolve schedule is described in [1].
    """

    def __init__(self, max_n, snap_in_ram, uf=1, ub=1, wd=2, rd=2):
        schedule = list(periodic_disk_revolve(max_n - 1, snap_in_ram, wd, rd, uf, ub))
        super().__init__(max_n, snap_in_ram, max_n - snap_in_ram, schedule)

class Revolve(RevolveCheckpointSchedule):
    """Revolve checkpointing schedule.

    Atributes
    ---------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snap_in_ram : int
        The maximum steps to store the forward checkpoints in RAM.
    uf : float
        The cost of advancing the forward over one step.
    ub : float
        The cost of advancing the adjoint over one step.
    wd : float
        The cost of writing the checkpoint data in disk.
    rd : float
        The cost of reading the checkpoint data from disk.

    Notes
    -----
    The H-Revolve schedule is described in [1].
    """

    def __init__(self, max_n, snap_in_ram, uf=1, ub=1, wd=2, rd=2):
        schedule = list(revolve(max_n - 1, snap_in_ram, wd, rd, uf, ub))
        super().__init__(max_n, snap_in_ram, max_n - snap_in_ram, schedule)

      
def _convert_action(action):
    """Convert an operation to a `checkpoint_schedules` action.

    Parameters
    ----------
    action_n : hrevolve_sequences.revolve.Operation
        An operation from the `checkpoint_schedules.hrevolve_sequences`.
    
    Notes
    -----
    The operations have `type` and `index` attributes. The `type` attribute is a string that gives
    the operation name, which are listed at the `checkpoint_schedules.schedule.basic_funtions.official_names` 
    dictionary. The `index` attribute is a tuple containing the time steps for some operations or the storage
    level and time step for other operations. For instance, if operation type is `Forward`, the operation index 
    that is a tuple `(n0, n1)`, which it read as the next action aims to execute the forward solver from the 
    step n0 to step n1. If the operation is `Write` type, the operation index is a tuple `(storage, n0)`.

    Returns
    -------
    str, tuple(int, int , str)
        Return the operation name, and a tuple containig the steps `n_0`, step `n_1` and the storage 
        level (either RAM or disk).

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
    elif cp_action in ["Read", "Write", "Discard"]:
        storage, n_0 = action.index
        n_1 = None
        storage = {0: StorageType.RAM, 1: StorageType.DISK}[storage]
    elif cp_action in ["Write_Forward", "Discard_Forward"]:
        _, n_0 = action.index
        n_1 = None
        storage = {0: StorageType.ADJ_DEPS}[0]
    elif cp_action in ["Write_Forward_memory",
                       "Discard_Forward_memory"]:
        n_0 = action.index
        n_1 = None
        storage = {0: StorageType.ADJ_DEPS}[0]
    elif cp_action in ["Read_disk", "Write_disk", "Discard_disk"]:
        n_0 = action.index
        n_1 = None
        storage = 1
        storage = {1: StorageType.DISK}[storage]
    elif cp_action in ["Read_memory", "Write_memory", "Discard_memory"]:
        n_0 = action.index
        n_1 = None
        storage = 0
        storage = {0: StorageType.RAM}[storage]
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