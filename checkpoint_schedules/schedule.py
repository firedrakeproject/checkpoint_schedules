#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import functools

__all__ = \
    [
        "CheckpointAction",
        "Forward",
        "Reverse",
        "Transfer",
        "EndForward",
        "EndReverse",
        "CheckpointSchedule",
    ]


class CheckpointAction:
    """Checkpoint action object.
    
    """
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args


class Forward(CheckpointAction):
    """Abstract forward action.

    Args
    ----
    n0 : int
        Initial step.
    n1 : int
        Final step.
    write_ics : bool
        This variable determine if the checkpont used to restart the forward
        solver will be written. If "True", the checkpoint at the time 'n0' is
        written.
    write_data : bool
        This variable determine if the checkpont used in the reverse 
        computation will be written. If "True", the checkpoint at the time 'n1' is
        written.
    storage : str
        Level of the checkpoint storage, either "RAM" or "disk".
    """
    def __init__(self, n0, n1, write_ics, write_data, storage):
        super().__init__(n0, n1, write_ics, write_data, storage)

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
            Initial step.
        """
        return self.args[0]

    @property
    def n1(self):
        """Final step of the forward solver.

        Returns
        -------
        int
            Final step.
        """
        return self.args[1]

    @property
    def write_ics(self):
        """Inform if the checkpoint at the step 'self.args[1]' is going
        to be saved.

        Returns
        -------
        bool
            If 'True', the checkpoint at the step 'self.args[1]' is going to be saved.
        """
        return self.args[2]
    
    @property
    def write_data(self):
        """Inform if the forward data at the step 'self.args[1]' is going
        to be saved.

        Returns
        -------
        bool
            If 'True', the forward data at the step 'self.args[1]' is going to be saved.
        """
        return self.args[3]
    
    @property
    def storage(self):
        """Level where the checkpoint is saved.

        Returns
        -------
        str
            Either "RAM" or "disk".
        """
        return self.args[4]
    
    def info(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        f_info = ("Forward(n0: " + str(self.n0) + ", n1: " + str(self.n1)
                  + ", write_ics: " + str(self.write_ics) + ", write_data: " + str(self.write_data)
                  + ", storage: " + str(self.storage) + ")")
        return f_info
    

class Reverse(CheckpointAction):
    """Reverse action.

    Attributes
    ----------
    n1 : int
        Initial step of reverse solver.
    n0 : int
        Final step of reverse solver.  
    clear_fwd_data : bool
        Determine if the forward data used in the reverse computation will be 
        cleaned. If "True", the intermediate forward data is cleaned.
    """
    def __init__(self, n1, n0, clear_fwd_data):

        super().__init__(n1, n0, clear_fwd_data)

    def __iter__(self):
        yield from range(self.n1 - 1, self.n0 - 1, -1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """Final step of the reverse mode.

        Returns
        -------
        float
            Final step.
        """
        return self.args[1]

    @property
    def n1(self):
        """Initial step of the reverse mode.

        Returns
        -------
        float
            Initial step.
        """
        return self.args[0]

    def clear_fwd_data(self):
        return self.args[2]
    
    def info(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        f_info = ("Reverse(n1: " + str(self.n1) + ", n0: " + str(self.n0)
                  + ", clear_fwd_data: " + str(self.clear_fwd_data()) + ")")
        return f_info

class Transfer(CheckpointAction):
    """Transfer from the snapshot to a space.
    """
    def __init__(self, n, from_storage, to_storage, delete=False):
        super().__init__(n, from_storage, to_storage, delete)

    @property
    def n(self):
        """Return the Write step.

        Returns
        -------
        int
            Write step.
        """
        return self.args[0]
    
    @property
    def from_storage(self):
        """Level where the checkpoint is saved.

        Returns
        -------
        str
            Either "RAM" or "disk".
        """
        return self.args[1]
    
    @property
    def to_storage(self):
        """Level where the checkpoint is saved.

        Returns
        -------
        str
            Either "RAM" or "disk".
        """
        return self.args[2]
    
    @property
    def delete(self):
        """Delete the checkpoint data saved in the snapshot.

        Returns
        -------
        bool
            Inform if the snapshot data will be deleted.
        """
        return self.args[3]
    
    def info(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        f_info = ("Transfer(n: " + str(self.n) + ", from_storage: " + str(self.from_storage) 
                  + ", to_storage: " + str(self.to_storage) + ", delete: " + str(self.delete) + ")")
        return f_info


class EndForward(CheckpointAction):
    """Action used if the Forward solver is ended.
    """
    def info(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        f_info = "EndForward"
        return f_info



class EndReverse(CheckpointAction):
    """Action used if the Reverse solver is ended.
    """
    def __init__(self, exhausted):
        super().__init__(exhausted)

    @property
    def exhausted(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        return self.args[0]
    
    def info(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        f_info = "EndReverse"
        return f_info

class CheckpointSchedule(ABC):
    """A checkpointing schedule.

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
        Maximal steps of a forward solver.
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
            """Checkpoint schedule Iterator.

            Returns
            -------
            object
                Iterator.
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
        """Abstract iterator.
        """
        raise NotImplementedError

    @abstractmethod
    def is_exhausted(self):
        """_summary_
        """
        raise NotImplementedError

    @abstractmethod
    def uses_disk_storage(self):
        """_summary_
        """
        raise NotImplementedError

    def n(self):
        """Forward step.

        Returns
        -------
        int 
            Forward step.
        """
        return self._n

    def r(self):
        """Reverse step.

        Returns
        -------
        int 
            Reverse step.
        """
        return self._r

    def max_n(self):
        """The number of forward steps in the initial forward calculation.

        Returns
        -------
        int
            The number of forward steps.
        """
        return self._max_n

    def is_running(self):
        return hasattr(self, "_iter")

    def finalize(self, n):
        """Verify if the solver is finalized.

        Parameters
        ----------
        n : int
            Current solver step.
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

