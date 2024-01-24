# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 The University of Edinburgh and Imperial College
# London
# Developed originally by James R. Maddison (j.r.maddison@ed.ac.uk).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).
import warnings
import functools
import numpy as np
from .schedule import CheckpointSchedule, Forward, Reverse, Copy, Move, \
    EndForward, EndReverse, StepType, StorageType

__all__ = ["MixedCheckpointSchedule"]

try:
    import numba
    from numba import njit
except ImportError:
    numba = None

    def njit(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapped_fn


class MixedCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule which mixes storage of forward restart data and
    non-linear dependency data in checkpointing units.

    Attributes
    ----------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snapshots: int
        The number of available checkpointing units.
    storage : StorageType
        Indicate the checkpointing unit storage location. Either `'RAM'` or
        `'disk'`.

    Notes
    -----
    Assumes that the data required to restart the forward has the same size as
    the data required to advance the adjoint over a step. Additionall details
    about the mixed checkpointing schedule is avaiable in [1].
    This is a offline checkpointing strategy, one adjoint calculation
    permitted.

    [1] Maddison, J. R. (2023). On the implementation of checkpointing with
    high-level algorithmic differentiation. arXiv preprint arXiv:2305.09568.
    DOI: 10.48550/arXiv.2305.09568
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
                        yield Forward(n0, n1 - 1, False, False, StorageType.WORK)  # noqa: E501
                    elif n1 <= n0:
                        raise InvalidForwardStep
                    self._n += 1
                    yield Forward(n1 - 1, n1, False, True, StorageType.WORK)  # noqa: E501
                elif step_type == StepType.FORWARD:
                    if n1 <= n0:
                        raise InvalidForwardStep
                    self._n = n1
                    yield Forward(n0, n1, False, False, StorageType.WORK)  # noqa: E501
                elif step_type == StepType.WRITE_ADJ_DEPS:
                    if n1 != n0 + 1:
                        raise InvalidForwardStep
                    self._n = n1
                    yield Forward(n0, n1, False, True, self._storage)
                    if n0 in snapshot_n:
                        raise RuntimeError("Invalid checkpointing state")
                    elif len(snapshots) > self._snapshots - 1:
                        raise RuntimeError("Invalid checkpointing state")
                    snapshot_n.add(n0)
                    snapshots.append((StepType.READ_ADJ_DEPS, n0))
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
            if step_type not in (StepType.FORWARD_REVERSE,
                                 StepType.READ_ADJ_DEPS):
                raise RuntimeError("Invalid checkpointing state")

            if self._r == 0:
                yield EndForward()

            self._r += 1
            yield Reverse(self._max_n - self._r + 1, self._max_n - self._r, True)  # noqa: E501

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
            if step_type == StepType.READ_ADJ_DEPS:
                # Non-linear dependency data checkpoint
                if not cp_delete:
                    # We cannot advance from a loaded non-linear dependency
                    # checkpoint, and so we expect to use it immediately
                    raise RuntimeError("Invalid checkpointing state")
                # Note that we cannot in general restart the forward here
                self._n += 1
            elif step_type != StepType.READ_ICS:
                raise RuntimeError("Invalid checkpointing state")
            if step_type == StepType.READ_ADJ_DEPS:
                storage_type = StorageType.WORK
            elif step_type == StepType.READ_ICS:
                storage_type = StorageType.WORK
            if cp_delete:
                yield Move(cp_n, self._storage, storage_type)
            else:
                yield Copy(cp_n, self._storage, storage_type)

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
        storage_type : StorageType
            Storage type to check.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        assert storage_type in StorageType
        return self._storage == storage_type


def cache_step(fn):
    _cache = {}

    @functools.wraps(fn)
    def wrapped_fn(n, s):
        # Avoid some cache misses
        s = min(s, n - 1)
        if (n, s) not in _cache:
            _cache[(n, s)] = fn(n, s)
        return _cache[(n, s)]

    return wrapped_fn


@cache_step
def optimal_steps_mixed(n, s):
    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n <= s + 1:
        return n
    elif s == 1:
        return n * (n + 1) // 2 - 1
    else:
        m = 1 + optimal_steps_mixed(n - 1, s - 1)
        for i in range(2, n):
            m = min(
                m,
                i
                + optimal_steps_mixed(i, s)
                + optimal_steps_mixed(n - i, s - 1))
        return m


@cache_step
def mixed_step_memoization(n, s):
    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n == 1:
        return (StepType.FORWARD_REVERSE, 1, 1)
    elif n <= s + 1:
        return (StepType.WRITE_ADJ_DEPS, 1, n)
    elif s == 1:
        return (StepType.WRITE_ICS, n - 1, n * (n + 1) // 2 - 1)
    else:
        m = None
        for i in range(2, n):
            m1 = (
                i
                + mixed_step_memoization(i, s)[2]
                + mixed_step_memoization(n - i, s - 1)[2])
            if m is None or m1 <= m[2]:
                m = (StepType.WRITE_ICS, i, m1)
        if m is None:
            raise RuntimeError("Failed to determine total number of steps")
        m1 = 1 + mixed_step_memoization(n - 1, s - 1)[2]
        if m1 <= m[2]:
            m = (StepType.WRITE_ADJ_DEPS, 1, m1)
        return m


_NONE = int(StepType.NONE)
_FORWARD = int(StepType.FORWARD)
_FORWARD_REVERSE = int(StepType.FORWARD_REVERSE)
_WRITE_ADJ_DEPS = int(StepType.WRITE_ADJ_DEPS)
_WRITE_ICS = int(StepType.WRITE_ICS)


@njit
def mixed_steps_tabulation(n, s):
    """Tabulate actions for a 'mixed' schedule, for the case where no forward
    restart checkpoint is stored at the start of the first step.

    Parameters
    ----------
    n : int
        The number of forward steps.
    s : int
        The number of checkpointing units.

    Returns
    -------
    ndarray
        Defines the schedule. `schedule[n_i, s_i, :]` indicates the action for
        the case of `n_i` steps and `s_i` checkpointing units. `schedule[n_i,
        s_i, 0]` defines the actions, `schedule[n_i, s_i, 1]` defines the
        number of forward steps to advance, and `schedule[n_i, s_i, 2]` defines
        the cost.
    """

    schedule = np.zeros((n + 1, s + 1, 3), dtype=np.int64)
    schedule[:, :, 0] = _NONE
    schedule[:, :, 1] = 0
    schedule[:, :, 2] = -1

    for s_i in range(s + 1):
        schedule[1, s_i, :] = (_FORWARD_REVERSE, 1, 1)
    for s_i in range(1, s + 1):
        for n_i in range(2, n + 1):
            if n_i <= s_i + 1:
                schedule[n_i, s_i, :] = (_WRITE_ADJ_DEPS, 1, n_i)
            elif s_i == 1:
                schedule[n_i, s_i, :] = (_WRITE_ICS, n_i - 1, n_i * (n_i + 1) // 2 - 1)  # noqa: E501
            else:
                for i in range(2, n_i):
                    assert schedule[i, s_i, 2] > 0
                    assert schedule[n_i - i, s_i - 1, 2] > 0
                    m1 = (
                        i
                        + schedule[i, s_i, 2]
                        + schedule[n_i - i, s_i - 1, 2])
                    if schedule[n_i, s_i, 2] < 0 or m1 <= schedule[n_i, s_i, 2]:  # noqa: E501
                        schedule[n_i, s_i, :] = (_WRITE_ICS, i, m1)
                if schedule[n_i, s_i, 2] < 0:
                    raise RuntimeError("Failed to determine total number of "
                                       "steps")
                assert schedule[n_i - 1, s_i - 1, 2] > 0
                m1 = 1 + schedule[n_i - 1, s_i - 1, 2]
                if m1 <= schedule[n_i, s_i, 2]:
                    schedule[n_i, s_i, :] = (_WRITE_ADJ_DEPS, 1, m1)
    return schedule


def cache_step_0(fn):
    _cache = {}

    @functools.wraps(fn)
    def wrapped_fn(n, s):
        # Avoid some cache misses
        s = min(s, n - 2)
        if (n, s) not in _cache:
            _cache[(n, s)] = fn(n, s)
        return _cache[(n, s)]

    return wrapped_fn


@cache_step_0
def mixed_step_memoization_0(n, s):
    if s < 0:
        raise ValueError("Invalid number of snapshots")
    if n < s + 2:
        raise ValueError("Invalid number of steps")

    if s == 0:
        return (StepType.FORWARD_REVERSE, n, n * (n + 1) // 2 - 1)
    else:
        m = None
        for i in range(1, n):
            m1 = (
                i
                + mixed_step_memoization(i, s + 1)[2]
                + mixed_step_memoization(n - i, s)[2])
            if m is None or m1 <= m[2]:
                m = (StepType.FORWARD, i, m1)
        if m is None:
            raise RuntimeError("Failed to determine total number of steps")
        return m


@njit
def mixed_steps_tabulation_0(n, s, schedule):
    """Tabulate actions for a 'mixed' schedule, for the case where a forward
    restart checkpoint is stored at the start of the first step.

    Parameters
    ----------
    n : int
        The number of forward steps.
    s : int
        The number of checkpointing units.
    schedule: ndarray
        As returned by `mixed_steps_tabulation`.

    Returns
    -------
    ndarray
        Defines the schedule. `schedule[n_i, s_i, :]` indicates the action for
        the case of `n_i` steps and `s_i` checkpointing units. `schedule[n_i,
        s_i, 0]` defines the actions, `schedule[n_i, s_i, 1]` defines the
        number of forward steps to advance, and `schedule[n_i, s_i, 2]` defines
        the cost.
    """

    schedule_0 = np.zeros((n + 1, s + 1, 3), dtype=np.int64)
    schedule_0[:, :, 0] = _NONE
    schedule_0[:, :, 1] = 0
    schedule_0[:, :, 2] = -1

    for n_i in range(2, n + 1):
        schedule_0[n_i, 0, :] = (_FORWARD_REVERSE, n_i, n_i * (n_i + 1) // 2 - 1)  # noqa: E501
    for s_i in range(1, s):
        for n_i in range(s_i + 2, n + 1):
            for i in range(1, n_i):
                assert schedule[i, s_i + 1, 2] > 0
                assert schedule[n_i - i, s_i, 2] > 0
                m1 = (
                    i
                    + schedule[i, s_i + 1, 2]
                    + schedule[n_i - i, s_i, 2])
                if schedule_0[n_i, s_i, 2] < 0 or m1 <= schedule_0[n_i, s_i, 2]:  # noqa: E501
                    schedule_0[n_i, s_i, :] = (_FORWARD, i, m1)
            if schedule_0[n_i, s_i, 2] < 0:
                raise RuntimeError("Failed to determine total number of "
                                   "steps")
    return schedule_0


class InvalidForwardStep(IndexError):
    "The forward step is not correct."


class InvalidReverseStep(IndexError):
    "The reverse step is not correct."


class InvalidRevolverAction(Exception):
    "The action is not expected for this iterator."


class InvalidActionIndex(IndexError):
    "The index of the action is not correct."
