#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains the checkpointing schedules for the cases where no revolver algorithm are used."""
import sys
from .schedule import CheckpointSchedule, Forward, Reverse,\
    EndForward, EndReverse, StorageType, Move, Copy


__all__ = \
    [
        "SingleMemoryStorageSchedule",
        "SingleDiskStorageSchedule",
        "NoneCheckpointSchedule",
        
    ]


class SingleMemoryStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all forward restart and non-linear
    dependency data are stored in memory.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    Parameters
    ----------
    storage : StorageType.RAM
        Indicate that the execution should stores in `'RAM'` all foward restart 
        data and non-linear dependency data.
        
    """

    def __init__(self, storage=StorageType.RAM):
        self._storage = storage
        super().__init__()
    
    def _iterator(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1, True, True, self._storage)

        yield EndForward()

        while True:
            if self._r == 0:
                # Reverse
                self._r = self._max_n
                yield Reverse(self._max_n, 0, True)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield EndReverse(False)
            else:
                raise RuntimeError("Invalid checkpointing state")

    @property
    def is_exhausted(self):
        return False

    def uses_storage_type(self, storage_type):
        """Check if the storage is in memory.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """

        return storage_type == self._storage


class SingleDiskStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all forward restart and non-linear
    dependency data are stored in disk.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    Parameters
    ----------
    storage : StorageType.WORKING_Disk
        Indicate that the execution should stores in memore all foward restart 
        data and non-linear dependency data.
        
    """

    def __init__(self, storage=StorageType.DISK, delete=False):
        self._storage = storage
        self._delete = delete
        super().__init__()
    
    def _iterator(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1, True, True, self._storage)

        yield EndForward()

        for i in range(self._max_n + 1):
            if self._r < self._max_n:
                # Reverse
                if self._delete:
                    Move(i, self._storage, StorageType.ADJ_DEPS)
                else:
                    Copy(i, self._storage, StorageType.ADJ_DEPS)

                yield Reverse(i + 1, i, True)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield EndReverse(False)
            else:
                raise RuntimeError("Invalid checkpointing state")
            self._r += 1

    @property
    def is_exhausted(self):
        return False

    def uses_storage_type(self, storage_type):
        """Check if the storage is in memory.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """

        return storage_type == self._storage


class NoneCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule for the case where no adjoint calculation is
    performed.

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
        """Check the storage type.

        Parameters
        ----------
        storage_type : StorageType
            Given storage type.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        return storage_type == StorageType.NONE

