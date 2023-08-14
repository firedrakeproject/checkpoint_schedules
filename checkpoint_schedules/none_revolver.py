#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module contains the checkpointing schedules for the cases where no 
revolver algorithm are used.
"""

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
    """A checkpointing schedule where all adjoint dependencies
    are stored in memory.

    Parameters
    ----------
    write_ics : bool
        Indicate whether to store the forward restart data for all steps.
    storage_ics: enum
        Indicate the storage type of the forward restart data. 
        This atributte is checked only if the user desires to save the forward 
        restart.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    `write_ics` is always `False` for this schedule by considering that storing
    the forward restart data is unnecessary by this schedule, as there is no
    need to recompute the forward solver while time advancing the adjoint
    solver.
    """

    def __init__(self):
        self._storage = StorageType.ADJ_DEPS
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
            yield Forward(n0, n1, False, True, StorageType.ADJ_DEPS)

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

        return storage_type == self._storage
    

class SingleDiskStorageSchedule(CheckpointSchedule):
    """A checkpointing schedule where all adjoint dependencies
    are stored in disk.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    Parameters
    ----------
    storage : enum
        Indicate that the execution should stores in memore all foward restart 
        data and non-linear dependency data.

    Notes
    -----
    Online, unlimited adjoint calculations permitted.

    `write_ics` is always `False` for this schedule by considering that storing
    the forward restart data is unnecessary by this schedule, as there is no
    need to recompute the forward solver while time advancing the adjoint
    solver.
    """

    def __init__(self, move_data=False):
        self._move_data = move_data
        self._storage = StorageType.ADJ_DEPS
        super().__init__()
    
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
                if self._move_data is True:
                    yield Move(i, StorageType.DISK, StorageType.ADJ_DEPS)
                else:
                    yield Copy(i, StorageType.DISK, StorageType.ADJ_DEPS)

                yield Reverse(i, i - 1, True)
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

    Parameters
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
        """Check the storage type.

        Parameters
        ----------
        storage_type : enum
            Storage type to check.

        Returns
        -------
        bool
            Whether this schedule uses the given storage type.
        """
        return storage_type == StorageType.NONE

