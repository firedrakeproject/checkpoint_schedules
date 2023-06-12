#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import functools
from enum import Enum
__all__ = \
    [
        "StorageLocation",
        "CheckpointAction",
        "Forward",
        "Reverse",
        "Copy",
        "EndForward",
        "EndReverse",
        "CheckpointSchedule",
    ]


class StorageLocation(Enum):
    """List of storage location.

    Notes
    -----
    "RAM" and "DISK" are referred to as storage levels that are
    chosen to store the checkpoiting data associated to different 
    step that were previously determined by the revolver schedule. 
    The checkpoint data are then copied to a location referred to as 
    TAPE that indicates the location where to store of the checkpoint 
    data used to restart the forward solver.
    If the checkpoint data stored in the two levels, "RAM" and "DISK", wont
    be copied again, they have the storage location NONE which means that will be deleted.
    """
    RAM = 0
    DISK = 1
    TAPE = -1
    NONE = None

class CheckpointAction:
    """Checkpoint action object.

    Parameters can be accessed via the `args` attribute.
    
    """
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"

    def __eq__(self, other):
        return isinstance(self, other) and self.args == other.args


class Forward(CheckpointAction):
    """The forward action which indicates the forward advancement.

    Aattributes
    -----------
    n0 : int
        Initial step of the forward computation.
    n1 : int
        Final step of the forward computation.
    write_ics : bool
        This variable indicare whether to store the checkpoint data 
        used to restart the forward solver. If "True", the checkpoint 
        at the time 'n0' is written is a storage level.
    adj_deps : bool
        This variable indicare whether to store the checkpont used in the 
        reverse computation. If "True", the checkpoint at the time 'n1' is
        written.
    storage : str
        Level of the checkpoint data storage, either "RAM" or "disk".

    """
    def __init__(self, n0, n1, write_ics, adj_deps, storage):
        super().__init__(n0, n1, write_ics, adj_deps, storage)

    def __iter__(self):
        yield from range(self.n0, self.n1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """Initial step of the forward solver.

        Returns
        -------
        int
            The initial step.
        """
        return self.args[0]

    @property
    def n1(self):
        """Final step of the forward solver.

        Returns
        -------
        int
            The final step.
        """
        return self.args[1]

    @property
    def write_ics(self):
        """Inform if the checkpoint at the step 'n' is going
        to be saved.

        Returns
        -------
        bool
            If 'True', the checkpoint at the step 'n0' is going to be saved.
        """
        return self.args[2]
    
    @property
    def adj_deps(self):
        """Inform if the forward data at the step 'n1' is going
        to be saved.

        Returns
        -------
        bool
            If 'True', the forward data at the step 'n1' is going to be saved.
        """
        return self.args[3]
    
    @property
    def storage(self):
        """Level where the checkpoint is saved.

        Notes
        -----
        The storage list are available at StorageLocation.

        See Also
        --------
        'StorageLocation'

        Returns
        -------
        str
            Either "RAM" or "disk".
        """
        return self.args[4]

class Reverse(CheckpointAction):
    """Reverse action which indicate the adjoint advancement.

    Attributes
    ----------
    n1 : int
        Initial step of adjoint solver.
    n0 : int
        Final step of adjoint solver.  
    clear_adj_deps : bool
        Indicate whether to clear the forward data used in the adjoint computation. 
        If "True" the forward data is cleaned.
    
    """
    def __init__(self, n1, n0, clear_adj_deps):

        super().__init__(n1, n0, clear_adj_deps)

    def __iter__(self):
        yield from range(self.n1 - 1, self.n0 - 1, -1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """Final step of the adjoint computation.

        Returns
        -------
        float
            The final step.
        """
        return self.args[1]

    @property
    def n1(self):
        """Initial step of the adjoint computation.

        Returns
        -------
        float
            The initial step.
        """
        return self.args[0]

    def clear_adj_deps(self):
        """Indicate whether to clear the forward data used in the reverse solver.

        Returns
        -------
        bool
            Return whether to clear the forward data.
        """
        return self.args[2]

class Copy(CheckpointAction):
    """Indicate the action of copying from a storage level to the TAPE. 

    
    Attributes
    ----------
    n : int
        The step with which the copied data is associated. 
    from_storage : str
        The storage from which the data should be copied. Either
        `'RAM'` or `'DISK'`. 
    to_storage : str
        The location to which the data should be copied. It is 
        referred to as TAPE.
    delete : bool
        Whether the data should be deleted from the indicated storage
        after it has been copied.

    Notes
    -----
        TAPE indicates the location of the checkpoint data used to 
        restart the forward solver.
        The storage levels are listed in the `StorageLocation`.
    """
    def __init__(self, n, from_storage, to_storage, delete=False):
        super().__init__(n, from_storage, to_storage, delete)

    @property
    def n(self):
        """The step where the copy action is employed.

        Returns
        -------
        int
            The copy step.
        """
        return self.args[0]
    
    @property
    def from_storage(self):
        """The storage level where it stores the checkpoint data.

        Notes
        -----
        Storage location are available in StorageLocation Class.

        
        See Also
        --------
        StorageLocation: Give a list of storage location.

        
        Returns
        -------
        str
            The storage levels, either "RAM" or "DISK".
        """
        return self.args[1]
    
    @property
    def to_storage(self):
        """The tape used to restart the foward solver.

        Returns
        -------
        str
            The "TAPE".
        """
        return self.args[2]
    
    @property
    def delete(self):
        """Delete the checkpoint saved in a storage level 
        that is informed at from_storage function.

        Returns
        -------
        bool
            If "True" the checkpoint saved in the storage level is deleted.
        """
        return self.args[3]


class EndForward(CheckpointAction):
    """Indicate that the forward solver is finalised.
    """



class EndReverse(CheckpointAction):
    """A checkpointing action which indicates the end of an adjoint
    calculation.

    Attributes
    ----------
    exhausted : bool
        Indicate whether the schedule has concluded.
        If `True` then this action should be the last action in the schedule.
    """
    def __init__(self, exhausted):
        super().__init__(exhausted)

    @property
    def exhausted(self):
        """Indicates whether the schedule has concluded.

        Returns
        -------
        bool
            If `True` then this action should be the last action in the schedule.
        """
        return self.args[0]
    

class CheckpointSchedule(ABC):
    """A checkpointing schedule.

    Notes
    -----
    The schedule is defined by iter, which yields actions in a similar manner
    to the approach used in
        A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
        of checkpointing for the reverse or adjoint mode of computational
        differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
        19--45, 2000
    e.g. 'forward', 'read', and 'write' correspond to ADVANCE, RESTORE, and
    TAKESHOT respectively in Griewank and Walther 2000 (although here 'write'
    actions occur *after* forward advancement from snapshots).

    The iter method yields (action, data), with:
    Forward(n0, n1)
    Run the forward from the start of block n0 to the start of block n1.

    Reverse(n1, n0)
    Run the adjoint from the start of block n1 to the start of block n0.

    Read(n, storage, delete)
    Read checkpoint data associated with block n from the indicated storage.
    delete indicates whether the checkpoint data should be deleted.

    Write(n, storage)
    Write checkpoint data associated with block n to the indicated storage.

    EndForward()
    End the forward calculation.

    EndReverse(exhausted)
    End a reverse calculation. If exhausted is False then a further reverse
    calculation can be performed.

    Atributes
    ----------
    max_n : int
        The number of forward steps in the initial forward calculation.
    """

    def __init__(self, max_n=None):
        if max_n is not None and max_n < 1:
            raise ValueError("max_n must be positive")

        self._n = 0
        self._r = 0
        self._max_n = max_n

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls_iter = cls.iter

        @functools.wraps(cls_iter)
        def iter(self):
            """Abstract checkpoint schedule iterator.

            Returns
            -------
            self._iter : generator
                Return an iterator object used to iterate the action given 
                by the revolve sequence schedules.
            """
            if not hasattr(self, "_iter"):
                self._iter = cls_iter(self)
            return self._iter

        cls.iter = iter

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter())

    @abstractmethod
    def iter(self):
        """A generator which should be overridden in derived classes in order
        to define a checkpointing schedule.
        """
        raise NotImplementedError

    @abstractmethod
    def is_exhausted(self):
        """Return whether the schedule has concluded. Note that some schedules
        permit multiple adjoint calculation, and may never conclude.
        """
        raise NotImplementedError

    @abstractmethod
    def uses_disk_storage(self):
        """Return whether the schedule may use disk storage. If `False` then no
        disk storage is required.
        """
        raise NotImplementedError

    def n(self):
        """Return the forward step location. 
        
        Notes
        -----
        After executing all actions defined so far in the schedule the forward 
        is at the start of this step.

        Returns
        -------
        int 
            The forward step location.
        """
        return self._n

    def r(self):
        """Return the reverse step that is the number of adjoint steps advanced.

        Returns
        -------
        self._r : int 
            The reverse step.
        """
        return self._r

    def max_n(self):
        """The number of forward steps in the initial forward calculation.

        Returns
        -------
        self._max_n : int
            The number of forward steps.
        """
        return self._max_n

    def is_running(self):
        """Return whether the schedule is 'running' -- i.e. at least one action
        has been defined so far in the schedule.

        Returns
        -------
        bool
            Indicate wheter the 'CheckpointSchedule' schedule has an iterator object. 
            If "True", the checkpoint schedules were not finalised.
        """
        return hasattr(self, "_iter")

    def finalize(self, n):
        """Indicate the number of forward steps in the initial forward
        calculation.

        Parameters
        ----------
        n : int
            The number of steps in the initial forward calculation.
        """
        if n < 1:
            raise ValueError("n must be positive")
        if self._max_n is None:
            if self._n >= n:
                self._n = n
                self._max_n = n
            else:
                raise RuntimeError("Invalid checkpointing state")
        elif self._n != n or self._max_n != n:
            raise RuntimeError("Invalid checkpointing state")


