#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""..."""
import sys
from .schedule import CheckpointSchedule, Forward, Reverse,\
    EndForward, EndReverse, StorageType


__all__ = \
    [
        "SingleStorageSchedule",
        "NoneCheckpointSchedule",
    ]


class SingleStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all forward restart and non-linear
    dependency data are stored.

    Online, unlimited adjoint calculations permitted.
    """

    def __init__(self, storage=StorageType.WORKING_MEMORY):
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
        """Check if a given storage type is used in this schedule.

        Parameters
        ----------
        storage_type : StorageType.RAM or StorageType.DISK
            Given storage type. Either :class:`StorageType.RAM` or
            :class:`StorageType.DISK`.

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
        storage_type : StorageType.RAM, StorageType.DISK or StorageType.NONE
            Given storage type.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        assert storage_type in StorageType
        return storage_type == StorageType.NONE

