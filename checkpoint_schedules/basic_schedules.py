"""This module contains the checkpointing schedules for the cases where no
revolver algorithm are used.
"""
import sys
from .schedule import CheckpointSchedule, Forward, Reverse, \
    EndForward, EndReverse, Move, Copy
from .utils import StorageType


__all__ = \
    [
        "SingleMemoryStorageSchedule",
        "SingleDiskStorageSchedule",
        "NoneCheckpointSchedule",
    ]


class SingleMemoryStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all adjoint dependencies
    are stored in memory.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.
    """

    def __init__(self):
        super().__init__()
        self._storage = StorageType.WORK

    def _iterator(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1, False, True, StorageType.WORK)

        yield EndForward()

        while True:
            if self._r == 0:
                # Reverse
                self._r = self._max_n
                yield Reverse(self._max_n, 0, True)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield EndReverse()
            else:
                raise RuntimeError("Invalid checkpointing state")

    @property
    def is_exhausted(self):
        return False

    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used in this schedule.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Notes
        -----
        This schedule uses only `StorageType.WORK`, which means to store all
        adjoint dependencies in a `'variable'` used for the adjoint
        computation. Thus, this method returns `True` only if
        `storage_type == StorageType.WORK`. Otherwise, it returns `False`.

        Returns
        -------
        bool
            Whether this schedule uses a given storage type.
        """

        return storage_type == self._storage


class SingleDiskStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all adjoint dependencies
    are stored on disk.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    Parameters
    ----------
    move_data : bool
        Indicate whether the execution should move the data from
        `StorageType.DISK` to `StorageType.WORK`.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.
    """

    def __init__(self, move_data=False):
        super().__init__()
        self._move_data = move_data
        self._storage = StorageType.DISK

    def _iterator(self):
        """Schedule iterator.
        """

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1, False, True, StorageType.DISK)

        yield EndForward()

        for i in range(self._max_n, 0, -1):
            if self._r < self._max_n:
                # Reverse
                if self._move_data:
                    yield Move(i, StorageType.DISK, StorageType.WORK)
                else:
                    yield Copy(i, StorageType.DISK, StorageType.WORK)

                yield Reverse(i, i - 1, True)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield EndReverse()
            else:
                raise RuntimeError("Invalid checkpointing state")
            self._r += 1

    @property
    def is_exhausted(self):
        if self._move_data:
            return True
        else:
            return False

    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used in this schedule.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Notes
        -----
        This schedule uses only `StorageType.DISK`, i.e, store all
        adjoint dependencies in `'disk'`. Thus, this method returns `True`
        only if `storage_type == StorageType.DISK`. Otherwise, it returns
        `False`.

        Returns
        -------
        bool
            Whether this schedule uses a given storage type.
        """

        return storage_type == self._storage


class NoneCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule for the case where no adjoint calculation is
    performed.

    Attributes
    ----------
    _exhausted : bool
        Indicate that the execution is exhausted.

    Notes
    -----
    Online, zero adjoint calculations permitted.
    """

    def __init__(self):
        super().__init__()
        self._exhausted = False

    def _iterator(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1, False, False, StorageType.NONE)

        self._exhausted = True
        yield EndForward()

    @property
    def is_exhausted(self):
        """Check if the execution is exhausted.
        """
        return self._exhausted

    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used in this schedule.

        Parameters
        ----------
        storage_type : enum
            Storage type to check.

        Notes
        -----
        This schedule is employed if there is no adjoint calculation,
        which leads no requirements for forward data (adjoint dependency)
        storage. Therefore, this method returns `True` only if
        `storage_type == StorageType.NONE`. Otherwise, it returns `False`.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        return storage_type == StorageType.NONE
