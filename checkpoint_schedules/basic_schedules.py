# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 The University of Edinburgh and Imperial College
# London
# Developed originally by James R. Maddison (j.r.maddison@ed.ac.uk).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains basic checkpointing schedules.
"""

import sys
from .schedule import (
    CheckpointSchedule, Forward, Reverse, EndForward, EndReverse, Move,
    Copy, StorageType)


__all__ = \
    [
        "SingleMemoryStorageSchedule",
        "SingleDiskStorageSchedule",
        "NoneCheckpointSchedule"
    ]


class SingleMemoryStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all adjoint dependencies are stored in
    memory.

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
        """Return whether a given storage type is used in this schedule.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Notes
        -----
        This schedule uses only `StorageType.WORK`.

        Returns
        -------
        bool
            Whether this schedule uses a given storage type.
        """

        return storage_type == self._storage


class SingleDiskStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all adjoint dependencies are stored on
    disk.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    Parameters
    ----------
    move_data : bool
        Indicate whether the execution should move the data from
        `StorageType.DISK` to `StorageType.WORK`, rather than copy the data.

    Notes
    -----
    Online, unlimited adjoint calculations permitted if `move_data` is `False`,
    one adjoint calculation permitted if `move_data` is `True`.
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
            n1 = n0 + 1
            self._n = n1
            yield Forward(n0, n1, False, True, StorageType.DISK)

        yield EndForward()

        while True:
            while self._r < self._max_n:
                n1 = self._max_n - self._r
                n0 = n1 - 1

                self._n = n0
                if self._move_data:
                    yield Move(self._n, StorageType.DISK, StorageType.WORK)
                else:
                    yield Copy(self._n, StorageType.DISK, StorageType.WORK)

                self._r = self._max_n - n0
                yield Reverse(n1, n0, True)
            if self._r > self._max_n:
                raise RuntimeError("Invalid checkpointing state")
            self._r = 0
            yield EndReverse()

            if self._move_data:
                break

    @property
    def is_exhausted(self):
        return (self._move_data
                and self._max_n is not None
                and self._r == self._max_n)

    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used by this schedule.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Notes
        -----
        This schedule uses only `StorageType.DISK` and `StorageType.WORK`.

        Returns
        -------
        bool
            Whether this schedule uses a given storage type.
        """

        return storage_type in {self._storage, StorageType.WORK}


class NoneCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule for the case where no adjoint calculation is
    performed.

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
        return self._exhausted

    def uses_storage_type(self, storage_type):
        """Check if a given storage type is used by this schedule.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Notes
        -----
        This schedule is employed if there is no adjoint calculation,
        which leads no requirements for forward data (adjoint dependency)
        storage. Therefore this method always returns `False`.

        Returns
        -------
        bool
            Whether this schedule uses a given storage type.
        """

        return False
