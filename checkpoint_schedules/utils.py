"""Utilities for checkpointing schedules."""

import functools
from enum import Enum
import numpy as np
from .schedule import StepType

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


class StorageType(Enum):
    """Storage types.

    RAM : Memory.

    DISK : Disk.

    WORK : Working memory location for the forward or adjoint.

    NONE : No storage. Used e.g. to indicate delete actions.

    Notes
    -----
    The data stored in `RAM` or on `DISK` should not be directly accessed by
    the forward or the adjoint, but should instead be copied or moved to `WORK`
    before usage.
    """

    RAM = 0
    DISK = 1
    WORK = -1
    NONE = None

    def __repr__(self):
        return type(self).__name__ + "." + self.name


@njit
def n_advance(n, snapshots, *, trajectory="maximum"):
    """Return the number of steps to advance in a Revolve schedule.

    Parameters
    ----------
    n : int
        The number of forward steps.
    snapshots : int
        The number of checkpointing units.
    trajectory : str, optional
        The trajectory to use. Can be `'maximum'` or `'revolve'`.

    Notes
    -----
    This function implements the algorithm described in [1].

    [1] Andreas Griewank and Andrea Walther, 'Algorithm 799: revolve: an
    implementation of checkpointing for the reverse or adjoint mode of
    computational differentiation', ACM Transactions on Mathematical
    Software, 26(1), pp. 19--45, 2000, doi: 10.1145/347837.347846.
    """

    if n < 1:
        raise ValueError("Require at least one block")
    if snapshots <= 0:
        raise ValueError("Require at least one snapshot")

    # Discard excess snapshots
    snapshots = max(min(snapshots, n - 1), 1)
    # Handle limiting cases
    if snapshots == 1:
        return n - 1  # Minimal storage
    elif snapshots == n - 1:
        return 1  # Maximal storage

    # Find t as in [1] Proposition 1 (note 'm' in [1] is 'n' here, and
    # 's' in [1] is 'snapshots' here). Compute values of beta as in equation
    # (1) of [1] as a side effect. We must have a minimal rerun of at least
    # 2 (the minimal rerun of 1 case is maximal storage, handled above) so we
    # start from t = 2.
    t = 2
    b_s_tm2 = 1
    b_s_tm1 = snapshots + 1
    b_s_t = ((snapshots + 1) * (snapshots + 2)) // 2
    while b_s_tm1 >= n or n > b_s_t:
        t += 1
        b_s_tm2 = b_s_tm1
        b_s_tm1 = b_s_t
        b_s_t = (b_s_t * (snapshots + t)) // t

    if trajectory == "maximum":
        # Return the maximal step size compatible with Fig. 4 of [1]
        b_sm1_tm2 = (b_s_tm2 * snapshots) // (snapshots + t - 2)
        if n <= b_s_tm1 + b_sm1_tm2:
            return n - b_s_tm1 + b_s_tm2
        b_sm1_tm1 = (b_s_tm1 * snapshots) // (snapshots + t - 1)
        b_sm2_tm1 = (b_sm1_tm1 * (snapshots - 1)) // (snapshots + t - 2)
        if n <= b_s_tm1 + b_sm2_tm1 + b_sm1_tm2:
            return b_s_tm2 + b_sm1_tm2
        elif n <= b_s_tm1 + b_sm1_tm1 + b_sm2_tm1:
            return n - b_sm1_tm1 - b_sm2_tm1
        else:
            return b_s_tm1
    elif trajectory == "revolve":
        # [1], equation at the bottom of p. 34
        b_sm1_tm1 = (b_s_tm1 * snapshots) // (snapshots + t - 1)
        b_sm2_tm1 = (b_sm1_tm1 * (snapshots - 1)) // (snapshots + t - 2)
        if n <= b_s_tm1 + b_sm2_tm1:
            return b_s_tm2
        elif n < b_s_tm1 + b_sm1_tm1 + b_sm2_tm1:
            return n - b_sm1_tm1 - b_sm2_tm1
        else:
            return b_s_tm1
    else:
        print(trajectory)
        raise ValueError("Unexpected trajectory: '{trajectory:s}'")


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


@cache_step
def optimal_extra_steps(n, s):
    """Return the optimal number of extra steps for binomial checkpointing.

    Parameters
    ----------
    n : int
        The number of forward steps.
    s : int
        The number of checkpointing units.

    Returns
    -------
    int
        The optimal number of extra steps.
    """

    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n == 1:
        return 0
    # Equation (2) of
    #   A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
    #   of checkpointing for the reverse or adjoint mode of computational
    #   differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
    #   19--45, 2000
    elif s == 1:
        return n * (n - 1) // 2
    else:
        m = None
        for i in range(1, n):
            m1 = (i
                  + optimal_extra_steps(i, s)
                  + optimal_extra_steps(n - i, s - 1))
            if m is None or m1 < m:
                m = m1
        if m is None:
            raise RuntimeError("Failed to determine number of extra steps")
        return m


def optimal_steps_binomial(n, s):
    """Return the optimal total number of steps for binomial checkpointing.

    Parameters
    ----------
    n : int
        The number of forward steps.
    s : int
        The number of checkpointing units.

    Returns
    -------
    int
        The optimal total number of steps.
    """

    return n + optimal_extra_steps(n, s)
