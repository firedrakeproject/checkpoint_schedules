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
        "Move",
        "EndForward",
        "EndReverse",
        "CheckpointSchedule",
    ]


class StorageType(Enum):
    """ Enums to indicate the storage type of the checkpoint data.

    RAM : A enum to indicate the storage of the forward checkpoint data in memory.

    DISK : A enum to indicate the storage of the forward checkpoint data in RAM.
    
    FWD_RESTART : A enum to indicate the storage of the forward checkpoint data
    in a "variable" used to initilise the forward solver.

    ADJ_DEPS : A enum to indicate the storage of the forward checkpoint data
    in a "variable" used for the adjoint computation.

    WORKING_MEMORY : A enum to indicate the storage of all forward checkpoint data
    is stored in memory.

    WORKING_DISK : A enum to indicate the storage of all forward checkpoint data
    is stored in disk.

    NONE : A enum to indicate that there is no specific storage location defined 
    for the checkpoint data.
    """
    RAM = 0
    DISK = 1
    FWD_RESTART = 2
    ADJ_DEPS = 3
    WORKING_MEMORY = 4
    WORKING_DISK = 5
    NONE = None


class StepType(IntEnum):
    """Enums to indicate which execution is being executed in such step.

    FORWARD : A enum to indicate the Forward execution is such step.

    FORWARD_REVERSE : A enum to indicate the foward and adjoint execution in such step.

    WRITE_ADJ_DEPS : A enum to indicate that the storage of the forward data required
    for the adjoint computation is such step.

    WRITE_ICS : A enum to indicate that the storage of the forward data required
    for the the forward solver restarting.

    READ_ADJ_DEPS : A enum to indicate that the read of the forward data required
    for the adjoint computation is such step.

    READ_ICS : A enum to indicate that the read of the forward data required
    for the the forward solver restarting.
    """
    NONE = 0
    FORWARD = 1
    FORWARD_REVERSE = 2
    WRITE_ADJ_DEPS = 3
    WRITE_ICS = 4
    READ_ADJ_DEPS = 5
    READ_ICS = 6


class CheckpointAction:
    """A checkpoint action base class.
    
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
    """This action indicates the forward advancement.
    This action also configures the intermediate storage.

    Attributes
    ----------
    n0 : int
        The forward should advance from the start of this step.
    n1 : int
        The forward should advance to the start of this step. 
    write_ics : bool
        Whether to store forward restart data. 
    write_adj_deps : bool
        Whether to store forward data required for the adjoint compuations. 
    storage : StorageType
        Indicates the storage type of the checkpoint data. 
    
    Notes
    -----
    To exemplify this action, let us consider a particular case:

    * `Forward(10, 25, True, False, StorageType.RAM)`
        This action is read as:
            - Advance the forward solver from the step 10 to the start of the step 25.

            - Write the forward data (`write_ics` is `'True'`) required to initialise
            a forward solver from the step 10.
            
            - It is not required to store the forward data for the adjoint 
            computation once `write_adj_deps` is False.

            - The forward data storage is in memory (`storage` is `StorageType.RAM`).

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
        return self.args[0]

    @property
    def n1(self):
        return self.args[1]

    @property
    def write_ics(self):
        return self.args[2]
    
    @property
    def write_adj_deps(self):
        return self.args[3]
    
    @property
    def storage(self):
        return self.args[4]


class Reverse(CheckpointAction):
    """This action indicates the adjoint advancement.

    Attributes
    ----------
    n1 : int
        The adjoint should advance from the start of this step. 
    n0 : int
        The adjoint should advance to the start of this step. 
    clear_adj_deps : bool
        Indicate whether to clear the forward data used for the adjoint
        computation. 
    
    Notes
    -----
    To exemplify this action, let us consider a particular case:

    * `Reverse(3, 2, True)`
        This action is read as:
            - Advance the adjoint solver from the step 3 to the start of the step 2.

            - Clear the forward data (`clear_adj_deps` is `'True'`) used for the adjoint
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
        return self.args[1]

    @property
    def n1(self):
        return self.args[0]

    @property
    def clear_adj_deps(self):
        return self.args[2]


class Copy(CheckpointAction):
    """Indicate the action of copying from a storage type
    to another storage type. 

    
    Attributes
    ----------
    n : int
        The step with which the copied data is associated.
    from_storage : StorageType
        Indicate the storage type from which the data should be copied. 
    to_storage : StorageType
        Indicate the storage type to which the data should be copied.

    Notes
    -----
    To exemplify this action, let us consider a particular cases:

    * `Copy(10, StorageType.RAM, StorageType.FWD_RESTART)`
        This action is read as:
            - Copy the forward checkpoint data of the step 10 which is
            stored in `StorageType.RAM` to the variable used for the 
            forward solver restarting.
    
    * `Copy(10, StorageType.DISK, StorageType.ADJ_DEPS)`
        This action is read as:
            - Copy the forward checkpoint data of the step 10 which is
            stored in `StorageType.DISK` to the variable used for the 
            adjoint computation.
    See Also
    --------
    :class:`StorageLocation` 

    """
    def __init__(self, n, from_storage, to_storage):
        super().__init__(n, from_storage, to_storage)

    @property
    def n(self):
        return self.args[0]
    
    @property
    def from_storage(self):
        return self.args[1]
    
    @property
    def to_storage(self):
        return self.args[2]


class Move(CheckpointAction):
    """Indicate the action of moving from a storage type
    to another storage type. 

    
    Attributes
    ----------
    n : int
        The step with which the copied data is associated.
    from_storage : StorageType
        Indicate the storage type from which the data should be copied. 
    to_storage : StorageType
        Indicate the storage type to which the data should be copied.
    
    Notes
    -----
    The difference between `Copy` and `Move` is that the `Move` action
    move the data from a storage type to another storage type. Hence,
    after the `Move` action, the data is not available in `from_storage`.
    On the other hand, the `Copy` action copy the data from a storage type 
    but the data is still available in `from_storage`.

    To exemplify this action, let us consider a particular case:

    * `Move(10, StorageType.RAM, StorageType.FWD_RESTART)`
        This action is read as:
            - Move the forward checkpoint data of the step 10 which is
            stored in `StorageType.RAM` to the variable used for the
            forward solver restarting. After this action, the forward
            checkpoint data of the step 10 is not available in 
            `StorageType.RAM`.


    """
    def __init__(self, n, from_storage, to_storage):
        super().__init__(n, from_storage, to_storage)

    @property
    def n(self):
        return self.args[0]

    @property
    def from_storage(self):
        return self.args[1]

    @property
    def to_storage(self):

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
        The number of steps in the initial forward calculation. If not
        supplied then this should later be provided by calling the
        :meth:`finalize` method.

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

    Schedules control an intermediate storage, which buffers forward restart
    data for forward restart checkpoints, and which stores non-linear
    dependency data either for storage in checkpointing units or for immediate
    use by the adjoint. The storage is accessed using the :class:`StorageType`.

    This schedule is able to execute in two modes: 'offline' and 'online'.
    In 'offline' schedules, where the number of steps in the forward
    calculation is initially known, this should be provided using the `max_n`
    argument on instantiation. In 'online' schedules, where the number of steps
    in the forward calculation is initially unknown, the number of forward
    steps should later be provided using the :meth:`finalize` method.
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
    def uses_storage_type(self, storage_type):
        """Return whether the schedule may use a type storage. 

        Parameters
        ----------
        storage_type : StorageType
            The storage type to check.
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


