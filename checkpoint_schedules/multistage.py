# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 The University of Edinburgh and Imperial College
# London
# Developed originally by James R. Maddison (j.r.maddison@ed.ac.uk).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).import functools
import functools
from operator import itemgetter
from .schedule import CheckpointSchedule, Forward, Reverse, Copy, Move, \
    EndForward, EndReverse, StorageType
from .mixed import cache_step

__all__ = \
    [
        "MultistageCheckpointSchedule",
        "optimal_steps_binomial"
    ]

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


def allocate_snapshots(max_n, snapshots_in_ram, snapshots_on_disk, *,
                       write_weight=1.0, read_weight=1.0, delete_weight=0.0,
                       trajectory="maximum"):
    """Allocate snapshots.

    Parameters
    ----------
    max_n : int
        The maximum number of forward steps in the calculation.
    snapshots_in_ram : int
        The maximum number of forward restart checkpoints to store in `'RAM'`.
    snapshots_on_disk : int
        The maximum number of forward restart checkpoints to store on `'disk'`.
    write_weight : float, optional
        The weight (cost?) of writing a checkpoint.
    read_weight : float, optional
        The weight (cost?) of a read from a checkpoint.
    delete_weight : float, optional
        The weight (cost?) of deleting a checkpoint.
    trajectory : str, optional
        The trajectory to use for allocating checkpoints.

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
    def action_copy(cp_action):
        nonlocal snapshot_i

        if snapshot_i < 0:
            raise RuntimeError("Invalid checkpointing state")
        weights[snapshot_i] += read_weight

    @action.register(Move)
    def action_move(cp_action):
        nonlocal snapshot_i

        if snapshot_i < 0:
            raise RuntimeError("Invalid checkpointing state")
        weights[snapshot_i] += read_weight
        if snapshot_i < 0:
            raise RuntimeError("Invalid checkpointing state")

        if cp_action.to_storage == StorageType.WORK or \
                cp_action.to_storage == StorageType.WORK:
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
    """A binomial checkpointing schedule.

    Attributes
    ----------
    max_n : int
        The number of forward steps in the initial forward calculation.
    snapshots_in_ram : int
        The maximum number of forward restart checkpoints to store in memory
        (`'RAM'`).
    snapshots_on_disk : int
        The maximum number of forward restart checkpoints to store on `'disk'`.
    trajectory : str
        When advancing `n` forward steps with `s` checkpointing
        units available there are in general multiple solutions to the problem
        of determining the number of forward steps to advance before storing
        a new forward restart checkpoint -- see Fig. 4 of [1]. This argument
        selects a solution:

            - `'revolve'`: The standard revolve solution, as specified in the
                equation at the bottom of p. 34 of GW2000.
            - `'maximum'`: The maximum possible number of steps, corresponding
                to the maximum step size compatible with the optimal region in
                Fig. 4 of GW2000.

    Notes
    -----
    This checkpointing approach is described in [1].
    Uses a 'MultiStage' distribution of checkpoints between `'RAM'` and
    `'disk'`
    equivalent to that described in [2].
    The distribution between RAM and disk is determined using an initial run of
    the schedule. Offline, one adjoint calculation permitted.

    The argument names `snapshots_in_ram` and `snapshots_on_disk` originate
    from the corresponding arguments for the :func:`adj_checkpointing`
    function in dolfin-adjoint (see e.g. version 2017.1.0).

    [1] Griewank, A., & Walther, A. (2000). Algorithm 799: revolve: an
    implementation of checkpointing for the reverse or adjoint mode of
    computational differentiation. ACM Transactions on Mathematical
    Software (TOMS), 26(1), 19-45., doi: 10.1145/347837.347846

    [2] Stumm, P., & Walther, A. (2009). Multistage approaches for optimal
    offline checkpointing. SIAM Journal on Scientific Computing, 31(3),
    1946-1967. doi: 10.1137/080718036
    """

    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 trajectory="maximum"):
        super().__init__(max_n=max_n)
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
        yield Forward(self._n - 1, self._n, False, True, StorageType.WORK)

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
                yield Move(cp_n, cp_storage, StorageType.WORK)
            else:
                self._n = cp_n
                yield Copy(cp_n, cp_storage, StorageType.WORK)
                n_snapshots = (self._snapshots_in_ram
                               + self._snapshots_on_disk
                               - len(snapshots) + 1)
                n0 = self._n
                n1 = n0 + n_advance(self._max_n - self._r - n0,
                                    n_snapshots,
                                    trajectory=self._trajectory)
                assert n1 > n0
                self._n = n1
                yield Forward(n0, n1, False, False, StorageType.WORK)

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
            yield Forward(self._n - 1, self._n, False, True, StorageType.WORK)  # noqa: E501
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
