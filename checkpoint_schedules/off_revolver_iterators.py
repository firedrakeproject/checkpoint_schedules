#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline checkpointing schedules iterator for the adjoint method."""
import functools
import warnings
from operator import itemgetter
from .schedule import CheckpointSchedule, Forward, Reverse, Copy,\
    EndForward, EndReverse, StorageType, StepType
from .hrevolve_sequences import hrevolve, disk_revolve, periodic_disk_revolve,\
      revolve
from .utils import convert_action, mixed_step_memoization,\
    mixed_step_memoization_0, mixed_steps_tabulation, mixed_steps_tabulation_0,\
    n_advance


__all__ = \
    [
        "HRevolve", "DiskRevolve", "PeriodicDiskRevolve",
        "MultistageCheckpointSchedule", "MixedCheckpointSchedule", 
        "Revolve"
    ]

try:
    import numba
except ImportError:
    numba = None


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
        Revolver sequence of operations.
    
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
        action, (n0, n1, storage)
            Schedule actions, step `n0`, step `n1` and the storage type.
        
        """
        if self._max_n is None:
            raise RuntimeError("Invalid forward steps number.")

        snapshots = set()
        w_storage = None
        write_ics = False
        adj_deps = False
        
        i = 0
        while i < len(self._schedule):
            cp_action, (n_0, n_1, storage) = convert_action(self._schedule[i])
            if cp_action == "Forward":
                if n_0 != self._n:
                    raise InvalidForwardStep
                self._n = n_1
                w_cp_action, (w_n0, _, w_storage) = convert_action(self._schedule[i - 1])
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
                    w_storage = StorageType.NONE
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
                    delete = True
                else:
                    delete = False
                yield Copy(n_0, storage, StorageType.TAPE, delete=delete)
            elif (cp_action == "Write" or cp_action == "Write_disk"
                  or cp_action == "Write_memory"):
                if n_0 != self._n:
                    raise InvalidActionIndex
            elif cp_action == "Write_Forward":
                if n_0 != self._n + 1:
                    raise InvalidActionIndex
                d_cp_action, (d_n0, _, w_storage) = convert_action(self._schedule[i + 3])
                if (d_cp_action != "Discard_Forward"
                    or d_n0 != n_0 or w_storage != storage):
                    if w_n0 != n_0:
                        raise InvalidActionIndex
                    write_ics = True
                    adj_deps = False
            elif cp_action == "Write_Forward_memory":
                if n_0 != self._n + 1:
                    raise InvalidActionIndex
                d_cp_action, (d_n0, _, w_storage) = convert_action(self._schedule[i + 3])
                if (d_cp_action != "Discard_Forward_memory"
                    or d_n0 != n_0 or w_storage != storage):
                    if w_n0 != n_0:
                        raise InvalidActionIndex
            elif cp_action == "Discard" or cp_action == "Discard_memory":
                if i < 2:
                    raise InvalidRevolverAction
                snapshots.remove(n_0)
            elif cp_action == "Discard_Forward" or cp_action == "Discard_Forward_memory":
                if n_0 != self._n:
                    raise InvalidActionIndex
            else:
                raise InvalidRevolverAction
            i += 1
        if len(snapshots) > self._snapshots_on_disk:
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
    """H-Revolve checkpointing schedule."""
    def __init__(self, max_n, snap_in_ram, snap_on_disk, fwd_cost=1, bwd_cost=1, w_cost=2, r_cost=2):
        cvec = (snap_in_ram, snap_on_disk)
        wc = [0, w_cost]
        rc = [0, r_cost]
        schedule = list(hrevolve(max_n - 1, cvec, wc, rc, fwd_cost, bwd_cost))
        super().__init__(max_n, snap_in_ram, snap_on_disk, schedule)
        

class DiskRevolve(RevolveCheckpointSchedule):
    """Disk Revolve checkpointing schedule.
    """

    def __init__(self, max_n, snap_in_ram, fwd_cost=1, bwd_cost=1, w_cost=2, r_cost=2):
        schedule = list(disk_revolve(max_n - 1, snap_in_ram, w_cost, r_cost, fwd_cost, bwd_cost))
        super().__init__(max_n, snap_in_ram, max_n - snap_in_ram, schedule)


class PeriodicDiskRevolve(RevolveCheckpointSchedule):
    """Periodic Disk Revolve checkpointing schedule."""

    def __init__(self, max_n, snap_in_ram, fwd_cost=1, bwd_cost=1, w_cost=2, r_cost=2):
        schedule = list(periodic_disk_revolve(max_n - 1, snap_in_ram, w_cost, r_cost, fwd_cost, bwd_cost))
        super().__init__(max_n, snap_in_ram, max_n - snap_in_ram, schedule)

class Revolve(RevolveCheckpointSchedule):
    """Revolve checkpointing schedule."""

    def __init__(self, max_n, snap_in_ram, fwd_cost=1, bwd_cost=1, w_cost=2, r_cost=2):
        schedule = list(revolve(max_n - 1, snap_in_ram, w_cost, r_cost, fwd_cost, bwd_cost))
        super().__init__(max_n, snap_in_ram, max_n - snap_in_ram, schedule)
        


def allocate_snapshots(max_n, snapshots_in_ram, snapshots_on_disk, *,
                       write_weight=1.0, read_weight=1.0, delete_weight=0.0,
                       trajectory="maximum"):
    """Allocate snapshots for a checkpointing schedule.
    
    Parameters
    ----------
    max_n : int
        The maximum number of forward steps in the calculation.
    snapshots_in_ram : int
        The maximum number of forward restart checkpoints to store in RAM.
    snapshots_on_disk : int
        The maximum number of forward restart checkpoints to store on disk.
    write_weight : float, optional
        The weight of a write to a checkpoint.
    read_weight : float, optional
        The weight of a read from a checkpoint.
    delete_weight : float, optional
        The weight of a delete of a checkpoint.
    trajectory : str, optional
        The trajectory to use for allocating checkpoints. See the `trajectory`.  
    """
    snapshots_in_ram = min(snapshots_in_ram, max_n - 1)
    snapshots_on_disk = min(snapshots_on_disk, max_n - 1)
    snapshots = min(snapshots_in_ram + snapshots_on_disk, max_n - 1)
    weights = [0.0 for _ in range(snapshots)]

    cp_schedule = MultistageCheckpointSchedule(max_n, snapshots, 0,
                                               trajectory=trajectory)

    snapshot_i = -1

    @functools.singledispatch
    def action(cp_action):
        raise TypeError(f"Unexpected checkpointing action: {cp_action}")

    @action.register(Copy)
    def action_read(cp_action):
        nonlocal snapshot_i

        if snapshot_i < 0:
            raise RuntimeError("Invalid checkpointing state")
        weights[snapshot_i] += read_weight
        if cp_action.delete:
            weights[snapshot_i] += delete_weight
            snapshot_i -= 1

    @action.register(Forward)
    def action_write(cp_action):
        nonlocal snapshot_i
        if cp_action.write_ics:
            snapshot_i += 1
            if snapshot_i >= snapshots:
                raise RuntimeError("Invalid checkpointing state")
            weights[snapshot_i] += write_weight

    @action.register(Copy)
    @action.register(Forward)
    @action.register(Reverse)
    @action.register(EndForward)
    @action.register(EndReverse)
    def action_pass(cp_action):
        pass

    # Run the schedule, keeping track of the total read/write/delete costs
    # associated with each storage location on the stack of checkpointing units

    while True:
        cp_action = next(cp_schedule)
        action(cp_action)
        if isinstance(cp_action, EndReverse):
            break

    assert snapshot_i == -1

    # Allocate the checkpointing units with highest cost to RAM, and the
    # remaining units to disk. For read and write costs of one and zero delete
    # costs the distribution of storage between RAM and disk is then equivalent
    # to that in
    #   Philipp Stumm and Andrea Walther, 'MultiStage approaches for optimal
    #   offline checkpointing', SIAM Journal on Scientific Computing, 31(3),
    #   pp. 1946--1967, 2009, doi: 10.1137/080718036

    allocation = [StorageType.DISK for _ in range(snapshots)]
    for i, _ in sorted(enumerate(weights), key=itemgetter(1),
                       reverse=True)[:snapshots_in_ram]:
        allocation[i] = StorageType.RAM

    return tuple(weights), tuple(allocation)


class MultistageCheckpointSchedule(CheckpointSchedule):
    """A binomial checkpointing schedule using the approach described in

      - Andreas Griewank and Andrea Walther, 'Algorithm 799: revolve: an
        implementation of checkpointing for the reverse or adjoint mode of
        computational differentiation', ACM Transactions on Mathematical
        Software, 26(1), pp. 19--45, 2000, doi: 10.1145/347837.347846

    hereafter referred to as GW2000.

    Uses a 'MultiStage' distribution of checkpoints between RAM and disk
    equivalent to that described in

        - Philipp Stumm and Andrea Walther, 'MultiStage approaches for optimal
          offline checkpointing', SIAM Journal on Scientific Computing, 31(3),
          pp. 1946--1967, 2009, doi: 10.1137/080718036

    The distribution between RAM and disk is determined using an initial run of
    the schedule.

    Offline, one adjoint calculation permitted.

    :arg max_n: The number of forward steps in the initial forward calculation.
    :arg snapshots_in_ram: The maximum number of forward restart checkpoints
        to store in memory.
    :arg snapshots_on_disk: The maximum number of forward restart checkpoints
        to store on disk.
    :arg trajectory: When advancing `n` forward steps with `s` checkpointing
        units available there are in general multiple solutions to the problem
        of determining the number of forward steps to advance before storing
        a new forward restart checkpoint -- see Fig. 4 of GW2000. This argument
        selects a solution:

            - `'revolve'`: The standard revolve solution, as specified in the
              equation at the bottom of p. 34 of GW2000.
            - `'maximum'`: The maximum possible number of steps, corresponding
              to the maximum step size compatible with the optimal region in
              Fig. 4 of GW2000.

    The argument names `snaps_in_ram` and `snaps_on_disk` originate from the
    corresponding arguments for the :func:`adj_checkpointing` function in
    dolfin-adjoint (see e.g. version 2017.1.0).
    """

    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 trajectory="maximum"):
        snapshots_in_ram = min(snapshots_in_ram, max_n - 1)
        snapshots_on_disk = min(snapshots_on_disk, max_n - 1)
        if snapshots_in_ram == 0:
            storage = tuple(StorageType.DISK for _ in range(snapshots_on_disk))
        elif snapshots_on_disk == 0:
            storage = tuple(StorageType.RAM for _ in range(snapshots_in_ram))
        else:
            _, storage = allocate_snapshots(
                max_n, snapshots_in_ram, snapshots_on_disk,
                trajectory=trajectory)

        snapshots_in_ram = storage.count(StorageType.RAM)
        snapshots_on_disk = storage.count(StorageType.DISK)

        super().__init__(max_n=max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._storage = storage
        self._exhausted = False
        self._trajectory = trajectory

    def _iterator(self):
        snapshots = []

        def write(n):
            if len(snapshots) >= self._snapshots_in_ram + self._snapshots_on_disk:  # noqa: E501
                raise RuntimeError("Unexpected snapshot number.")
            snapshots.append(n)
            return self._storage[len(snapshots) - 1]

        # Forward
        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")
        while self._n < self._max_n - 1:
            n_snapshots = (self._snapshots_in_ram
                           + self._snapshots_on_disk
                           - len(snapshots))
            n0 = self._n
            n1 = n0 + n_advance(self._max_n - n0, n_snapshots,
                                trajectory=self._trajectory)
            assert n1 > n0
            self._n = n1
            cp_storage = write(n0)
            yield Forward(n0, n1, True, False, cp_storage)

        if self._n != self._max_n - 1:
            raise RuntimeError("Invalid checkpointing state")

        # Forward -> reverse
        self._n += 1
        yield Forward(self._n - 1, self._n, False, True, StorageType.TAPE)

        yield EndForward()

        self._r += 1
        yield Reverse(self._n, self._n - 1, True)

        # Reverse
        while self._r < self._max_n:
            if len(snapshots) == 0:
                raise RuntimeError("Invalid checkpointing state")
            cp_n = snapshots[-1]
            cp_storage = self._storage[len(snapshots) - 1]
            if cp_n == self._max_n - self._r - 1:
                snapshots.pop()
                self._n = cp_n
                yield Copy(cp_n, cp_storage, True)
            else:
                self._n = cp_n
                yield Copy(cp_n, cp_storage, False)
                n_snapshots = (self._snapshots_in_ram
                               + self._snapshots_on_disk
                               - len(snapshots) + 1)
                n0 = self._n
                n1 = n0 + n_advance(self._max_n - self._r - n0,
                                    n_snapshots,
                                    trajectory=self._trajectory)
                assert n1 > n0
                self._n = n1
                yield Forward(n0, n1, False, False, StorageType.NONE)

                while self._n < self._max_n - self._r - 1:
                    n_snapshots = (self._snapshots_in_ram
                                   + self._snapshots_on_disk
                                   - len(snapshots))
                    n0 = self._n
                    n1 = n0 + n_advance(self._max_n - self._r - n0,
                                        n_snapshots,
                                        trajectory=self._trajectory)
                    assert n1 > n0
                    self._n = n1
                    cp_storage = write(n0)
                    yield Forward(n0, n1, True, False, cp_storage)

                if self._n != self._max_n - self._r - 1:
                    raise RuntimeError("Invalid checkpointing state")
                
            self._n += 1
            yield Forward(self._n - 1, self._n, False, True, StorageType.TAPE)
            self._r += 1
            yield Reverse(self._n, self._n - 1, True)
        if self._r != self._max_n:
            raise RuntimeError("Invalid checkpointing state")
        if len(snapshots) != 0:
            raise RuntimeError("Invalid checkpointing state")

        self._exhausted = True
        yield EndReverse()

    @property
    def is_exhausted(self):
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
 

class MixedCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule which mixes storage of forward restart data and
    non-linear dependency data in checkpointing units. Assumes that the data
    required to restart the forward has the same size as the data required to
    advance the adjoint over a step.

    Described in

        - James R. Maddison, 'On the implementation of checkpointing with
          high-level algorithmic differentiation',
          https://arxiv.org/abs/2305.09568v1, 2023

    Offline, one adjoint calculation permitted.

    :arg max_n: The number of forward steps in the initial forward calculation.
    :arg snapshots: The number of available checkpointing units.
    :arg storage: Checkpointing unit storage location. Either `'RAM'` or
        `'disk'`.
    """

    def __init__(self, max_n, snapshots, *, storage=StorageType.DISK):
        if snapshots < min(1, max_n - 1):
            raise ValueError("Invalid number of snapshots")
        if storage not in [StorageType.RAM, StorageType.DISK]:
            raise ValueError("Invalid storage")

        super().__init__(max_n)
        self._exhausted = False
        self._snapshots = min(snapshots, max_n - 1)
        self._storage = storage

    def _iterator(self):
        snapshot_n = set()
        snapshots = []

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")

        if numba is None:
            warnings.warn("Numba not available -- using memoization",
                          RuntimeWarning)
        else:
            schedule = mixed_steps_tabulation(self._max_n, self._snapshots)
            schedule_0 = mixed_steps_tabulation_0(self._max_n, self._snapshots, schedule)  # noqa: E501

        step_type = StepType.NONE
        while True:
            while self._n < self._max_n - self._r:
                n0 = self._n
                if n0 in snapshot_n:
                    # n0 checkpoint exists
                    if numba is None:
                        step_type, n1, _ = mixed_step_memoization_0(
                            self._max_n - self._r - n0,
                            self._snapshots - len(snapshots))
                    else:
                        step_type, n1, _ = schedule_0[
                            self._max_n - self._r - n0,
                            self._snapshots - len(snapshots)]
                else:
                    # n0 checkpoint does not exist
                    if numba is None:
                        step_type, n1, _ = mixed_step_memoization(
                            self._max_n - self._r - n0,
                            self._snapshots - len(snapshots))
                    else:
                        step_type, n1, _ = schedule[
                            self._max_n - self._r - n0,
                            self._snapshots - len(snapshots)]
                n1 += n0

                if step_type == StepType.FORWARD_REVERSE:
                    if n1 > n0 + 1:
                        self._n = n1 - 1
                        yield Forward(n0, n1 - 1, False, False, StorageType.NONE)
                    elif n1 <= n0:
                        raise InvalidForwardStep
                    self._n += 1
                    yield Forward(n1 - 1, n1, False, True, StorageType.TAPE)
                elif step_type == StepType.FORWARD:
                    if n1 <= n0:
                        raise InvalidForwardStep
                    self._n = n1
                    yield Forward(n0, n1, False, False, StorageType.NONE)
                elif step_type == StepType.WRITE_DATA:
                    if n1 != n0 + 1:
                        raise InvalidForwardStep
                    self._n = n1
                    yield Forward(n0, n1, False, True, self._storage)
                    if n0 in snapshot_n:
                        raise RuntimeError("Invalid checkpointing state")
                    elif len(snapshots) > self._snapshots - 1:
                        raise RuntimeError("Invalid checkpointing state")
                    snapshot_n.add(n0)
                    snapshots.append((StepType.READ_DATA, n0))
                elif step_type == StepType.WRITE_ICS:
                    if n1 <= n0 + 1:
                        raise InvalidActionIndex
                    self._n = n1
                    yield Forward(n0, n1, True, False, self._storage)
                    if n0 in snapshot_n:
                        raise RuntimeError("Invalid checkpointing state")
                    elif len(snapshots) > self._snapshots - 1:
                        raise RuntimeError("Invalid checkpointing state")
                    snapshot_n.add(n0)
                    snapshots.append((StepType.READ_ICS, n0))
                else:
                    raise RuntimeError("Unexpected step type")
            if self._n != self._max_n - self._r:
                raise InvalidForwardStep
            if step_type not in (StepType.FORWARD_REVERSE, StepType.READ_DATA):
                raise RuntimeError("Invalid checkpointing state")

            if self._r == 0:
                yield EndForward()

            self._r += 1
            yield Reverse(self._max_n - self._r + 1, self._max_n - self._r, True)

            if self._r == self._max_n:
                break

            step_type, cp_n = snapshots[-1]

            # Delete if we have (possibly after deleting this checkpoint)
            # enough storage left to store all non-linear dependency data
            cp_delete = (cp_n >= (self._max_n - self._r - 1
                                  - (self._snapshots - len(snapshots) + 1)))
            if cp_delete:
                snapshot_n.remove(cp_n)
                snapshots.pop()

            self._n = cp_n
            if step_type == StepType.READ_DATA:
                # Non-linear dependency data checkpoint
                if not cp_delete:
                    # We cannot advance from a loaded non-linear dependency
                    # checkpoint, and so we expect to use it immediately
                    raise RuntimeError("Invalid checkpointing state")
                # Note that we cannot in general restart the forward here
                self._n += 1
            elif step_type != StepType.READ_ICS:
                raise RuntimeError("Invalid checkpointing state")
            yield Copy(cp_n, self._storage, cp_delete)

        if len(snapshot_n) > 0 or len(snapshots) > 0:
            raise RuntimeError("Invalid checkpointing state")

        self._exhausted = True
        yield EndReverse()

    @property
    def is_exhausted(self):
        return self._exhausted

    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used in this schedule.

        Parameters
        ----------
        storage_type : StorageType.RAM or StorageType.DISK
            Given storage type.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        assert storage_type in StorageType
        return self._storage == storage_type

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