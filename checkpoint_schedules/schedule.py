#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import functools

__all__ = \
    [
        "CheckpointAction",
        "Forward",
        "Reverse",
        "Read",
        "Write",
        "WriteForward",
        "Clear",
        "Configure",
        "EndForward",
        "EndReverse",
        "CheckpointSchedule"
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

class Clear(CheckpointAction):
    """Clear checkpoint data.

    Args
    ----
    clear_ics : bool
        If "True", the checkpoint used to restart the forward solver is comming clear.
    clear_data : bool
        If True, the latest forward checkpoint data is coming clear.
    """
    def __init__(self, clear_ics, clear_data):
        self.type = "Clear"
        super().__init__(clear_ics, clear_data)

    @property
    def clear_ics(self):
        return self.args[0]

    @property
    def clear_data(self):
        return self.args[1]


class Configure(CheckpointAction):
    """Configure the type of checkpoint that is being saved.

    Args
    ----
    store_ics : bool
        If "True", the checkpoint data is coming save at the step give by 'Write' action.
    store_data : bool
        If "True", the latest checkpoint data is coming save.

    """
    def __init__(self, store_ics, store_data):
        self.type = "Configure"
        super().__init__(store_ics, store_data)

    @property
    def store_ics(self):
        """

        Returns
        -------
        _type_
            _description_
        """
        return self.args[0]

    @property
    def store_data(self):
        return self.args[1]


class Forward(CheckpointAction):
    """Abstract forward action.

    Args
    ----
    n0 : int
        initial step.
    n1 : int
        Final step.
    """
    def __init__(self, n0, n1):
        self.type = "Forward"
        super().__init__(n0, n1)

    def __iter__(self):
        yield from range(self.n0, self.n1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """Initial step of the forward mode.

        Returns
        -------
        float
            Initial step.
        """
        return self.args[0]

    @property
    def n1(self):
        """Final step of the forward mode.

        Returns
        -------
        float
            Final step.
        """
        return self.args[1]


class Reverse(CheckpointAction):
    """Reverse action.

    Attributes
    ----------
    n1 : int
        Initial step of reverse solver.
    n0 : int
        Final step of reverse solver.  
    """
    def __init__(self, n1, n0):
        self.type = "Reverse"
        super().__init__(n1, n0)

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


class Read(CheckpointAction):
    """Action read a checkpoint.
    """
    def __init__(self, n, storage, delete):
        self.type = "Read"
        super().__init__(n, storage, delete)

    @property
    def n(self):
        """Step.

        Returns
        -------
        int
            Current step.
        """
        return self.args[0]

    @property
    def storage(self):
        """Checkpoint storage.

        Returns
        -------
        bool
            ???.
        """
        return self.args[1]

    @property
    def delete(self):
        """Delete.

        Returns
        -------
        bool
            If "True", the checkpoint data is deleted.
        """
        return self.args[2]


class Write(CheckpointAction):
    
    def __init__(self, n, storage):
        self.type = "Write"
        super().__init__(n, storage)

    @property
    def n(self):
        """Step.

        Returns
        -------
        int
            Current step.
        """
        return self.args[0]

    @property
    def storage(self):
        """Checkpoint write.

        Returns
        -------
        bool
            If "True", the checkpoint data at a step `n` is saved.
        """
        return self.args[1]

class WriteForward(CheckpointAction):
    
    def __init__(self, n, storage):
        self.type = "WriteForward"
        super().__init__(n, storage)

    @property
    def n(self):
        """Step.

        Returns
        -------
        int
            Current step.
        """
        return self.args[0]

    @property
    def storage(self):
        """Checkpoint write.

        Returns
        -------
        bool
            If "True", the checkpoint data at a step `n` is saved.
        """
        return self.args[1]

class EndForward(CheckpointAction):
    """End forward action.
    """
    def __init__(self):
        self.type = "EndForward"


class EndReverse(CheckpointAction):
    """End reverse action.
    """
    def __init__(self, exhausted):
        self.type = "EndReverse"
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
        return self._n

    def r(self):
        return self._r

    def max_n(self):
        return self._max_n

    def is_running(self):
        return hasattr(self, "_iter")

    def finalize(self, n):
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
