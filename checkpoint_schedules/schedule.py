#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import functools
from enum import Enum, IntEnum

__all__ = \
    [
        "StorageType",
        "CheckpointAction",
        "Forward",
        "Reverse",
        "Copy",
        "EndForward",
        "EndReverse",
        "CheckpointSchedule",
    ]


class StorageType(Enum):
    """
    RAM : It is the first level of checkpoint storage.

    DISK : It is the second level of checkpoint storage.

    NONE : Indicate that there is no specific storage location defined 
    for the checkpoint data.
    """
    RAM = 0
    DISK = 1
    TAPE = -1
    NONE = None


class StepType(IntEnum):
    """Step type cost.
    """
    NONE = 0
    FORWARD = 1
    FORWARD_REVERSE = 2
    WRITE_DATA = 3
    WRITE_ICS = 4
    READ_DATA = 5
    READ_ICS = 6


class CheckpointAction:
    """A checkpoint action.
    
    Attributes
    ----------
    *args : Any
        The *args correspond to the arguments of the checkpoint scheduled actions: 
        `Forward`, `Reverse`, `Copy`, `EndForward`, `EndReverse`.
    
    See Also
    --------
    :class:`Forward`, :class:`Reverse`, :class:`Copy`, :class:`EndForward`,
    :class:`EndReverse`.
    
    """
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"

    def __eq__(self, other):
        return isinstance(self, other) and self.args == other.args


class Forward(CheckpointAction):
    """This action indicates the advancement of the forward solver 
    and configures the intermediate storage.

    Attributes
    ----------
    n0 : int
        Initial step of the forward computation.
    n1 : int
        Final step of the forward computation.
    write_ics : bool
        Indicate whether to store the checkpoint data used 
        to restart the forward solver.
    write_adj_deps : bool
        Indicate whether to store the checkpont data used in 
        the reverse computation.
    storage : str
        Indicate the level to store the checkpoint data, which can be either `RAM` or `DISK`.

    
    Notes
    -----
    The Forward action also indicates wheter to store the forward checkpoint data 
    used either to restart the forward solver or used in the adjoint computator.
    To exemplify, let us consider a particular case:
    Forward(0, 3, True, False, 'RAM'):
    This action is read as:
        - Execute the forward solver from step 0 to step 3.

        - Write the forward data (*write_ics*) of step 0 to RAM (storage).

        - It is not required to store the forward data for the adjoint 
        computation since *write_adj_deps* is False.

    """
    def __init__(self, n0, n1, write_ics, write_adj_deps, storage):
        super().__init__(n0, n1, write_ics, write_adj_deps, storage)

    def __iter__(self):
        yield from range(self.n0, self.n1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """Initial step of the forward computation.

        Returns
        -------
        int
            The initial step.
        """
        return self.args[0]

    @property
    def n1(self):
        """Final step of the forward computation.

        Returns
        -------
        int
            The final step.
        """
        return self.args[1]

    @property
    def write_ics(self):
        """Indicate whether to store the checkpoint data.

        Returns
        -------
        bool
            Write the forward data of step n0 if ``True``.
        """
        return self.args[2]
    
    @property
    def write_adj_deps(self):
        """Indicate wheter to store the forward data at step `n1`.

        Returns
        -------
        bool
            The forward data at the step `n1` is going to be saved if ``True``.
        """
        return self.args[3]
    
    @property
    def storage(self):
        """Level to store the checkpoint data.

        Notes
        -----
        The storage location list are available at `StorageType`.

        See Also
        --------
        :class:`StorageType`.

        Returns
        -------
        str
            Either :class:`StorageType.RAM` or :class:`StorageType.DISK`.
        """
        return self.args[4]


class Reverse(CheckpointAction):
    """This checkpoint action indicates the adjoint advancement.

    Attributes
    ----------
    n1 : int
        Initial step of adjoint solver.
    n0 : int
        Final step of adjoint solver.  
    clear_adj_deps : bool
        Indicate whether to clear the forward data used in the adjoint
        computation. 
    
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
        int
            The final step.
        """
        return self.args[1]

    @property
    def n1(self):
        """Initial step of the adjoint computation.

        Returns
        -------
        int
            The initial step.
        """
        return self.args[0]

    @property
    def clear_adj_deps(self):
        """Indicate whether to clear the forward data used in the reverse
        solver.

        Returns
        -------
        bool
            Clear the forward data used in the reverse computation if ``True``.
        """
        return self.args[2]


class Copy(CheckpointAction):
    """Indicate the action of copying from a storage level. 

    
    Attributes
    ----------
    n : int
        The step with which the copied data is associated.
    from_storage : str
        The storage level from which the data should be copied. Either
        `StorageType.RAM` or `StorageType.DISK`. 
    delete : bool
        Whether the data should be deleted from the indicated storage level
        after it has been copied.
    
    See Also
    --------
    :class:`StorageType` 

    """
    def __init__(self, n, from_storage, delete):
        super().__init__(n, from_storage, delete)

    @property
    def n(self):
        """The step to copy the forward checkpoint data.

        Returns
        -------
        int
            The copy step.
        """
        return self.args[0]
    
    @property
    def from_storage(self):
        """The storage type to copy the checkpoint data.

        Notes
        -----
        Storage type are available in `StorageType`.

        
        See Also
        --------
        :class:`StorageType`

        
        Returns
        -------
        str
            Either :class:`StorageType.RAM` or `StorageType.RAM`.
        """
        return self.args[1]
    
    @property
    def delete(self):
        """Delete the checkpoint stored in a storage type. 
        

        Notes
        -----
        The storage level is given by `from_storage` property.

        Returns
        -------
        bool
            Delete the checkpoint data if ``True``.
        """
        return self.args[2]


class EndForward(CheckpointAction):
    """Indicate that the forward solver is finalised.
    """


class EndReverse(CheckpointAction):
    """A checkpointing action which indicates the end of an adjoint
    calculation.
    """
    

class CheckpointSchedule(ABC):
    """A checkpointing schedule.

    Attributes
    ----------
    max_n : int
        The number of forward steps in the initial forward calculation.

    Notes
    -----
    Actions in the schedule are accessed by iterating over elements, and
    actions may be implemented using single-dispatch functions. e.g.

    .. code-block:: python

        @functools.singledispatch
        def action(cp_action):
            raise TypeError(f"Unexpected checkpointing action: {cp_action}")

        @action.register(Forward)
        def action_forward(cp_action):
            logger.debug(f"forward: forward advance to {cp_action.n1:d}")

        # ...

        for cp_action in cp_schedule:
            action(cp_action)
            if isinstance(cp_action, EndReverse):
                break
    """

    def __init__(self, max_n=None):
        if max_n is not None and max_n < 1:
            raise ValueError("max_n must be positive")

        self._n = 0
        self._r = 0
        self._max_n = max_n

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls_iter = cls._iterator

        @functools.wraps(cls_iter)
        def _iterator(self):
            """Abstract checkpoint schedule iterator.

            Returns
            -------
            self._iter : generator
                Return an iterator object used to iterate the action that is 
                given by the revolve schedule.
            """
            if not hasattr(self, "_iter"):
                self._iter = cls_iter(self)
            return self._iter

        cls._iterator = _iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator())

    @abstractmethod
    def _iterator(self):
        """A generator which should be overridden in derived classes in order
        to define a checkpointing schedule.
        """
        raise NotImplementedError

    @property
    def is_exhausted(self):
        """Return whether the schedule has concluded. 
        
        Notes
        -----
        Note that some schedules permit multiple adjoint calculation, 
        and may never conclude.
        """
        raise NotImplementedError

    @abstractmethod
    def uses_storage_type(self):
        """Return whether the schedule may use disk storage. 
        """
        raise NotImplementedError

    @property
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

    @property
    def r(self):
        """Return the reverse step.

        Notes
        -----
        The reverse step means the number of adjoint steps advancement.

        Returns
        -------
        self._r : int 
            The reverse step.
        """
        return self._r
    
    @property
    def max_n(self):
        """The number of forward steps in the initial forward calculation.

        Returns
        -------
        self._max_n : int
            The number of forward steps.
        """
        return self._max_n
    
    @property
    def is_running(self):
        """Return whether the schedule is `running`.

        Returns
        -------
        bool
            Indicate whether the :class:`CheckpointSchedule` schedule has an 
            iterator object. Do not finalise the checkpoint schedule if ``True``.
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


