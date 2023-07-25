import warnings
from .schedule import CheckpointSchedule, Forward, Reverse, Copy,\
    EndForward, EndReverse, StorageType, StepType

from .utils import mixed_step_memoization, mixed_step_memoization_0,\
     mixed_steps_tabulation, mixed_steps_tabulation_0

try:
    import numba
except ImportError:
    numba = None

__all__ = ["MixedCheckpointSchedule"]

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
            yield Copy(cp_n, self._storage, StorageType.TAPE, delete=cp_delete)

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