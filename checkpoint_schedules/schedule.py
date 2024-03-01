# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 The University of Edinburgh and Imperial College
# London
# Developed by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk),
# James R. Maddison (j.r.maddison@ed.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""Classes used to define checkpointing schedules and actions in checkpointing
schedules.
"""

from abc import ABC, abstractmethod
import functools
from enum import IntEnum, Enum
import sys

__all__ = \
    [
        "CheckpointAction",
        "Forward",
        "Reverse",
        "Copy",
        "Move",
        "EndForward",
        "EndReverse",
        "CheckpointSchedule",
        "StorageType",
    ]


class StorageType(Enum):
    """Storage types.

    RAM : Memory.

    DISK : Disk.

    WORK : Working memory location for the forward or adjoint.

    NONE : No storage. Used e.g. to indicate delete actions.

    Notes
    -----
    The data stored in `RAM` or on `DISK` should not be directly accessed by
    the forward or the adjoint, but should instead be copied or moved to `WORK`
    before usage.
    """

    RAM = 0
    DISK = 1
    WORK = -1
    NONE = None

    def __repr__(self):
        return type(self).__name__ + "." + self.name


class StepType(IntEnum):
    """Used when generating schedules, particularly when solving a dynamic
    programming problem, to indicate the next actions to perform.

    `FORWARD` : Forward advancement with no forward data saved.

    `FORWARD_REVERSE` : Forward advancement, followed by adjoint advancement of
    one step.

    `WRITE_ADJ_DEPS` : Forward advancement with saving of forward data required
    to advance the adjoint.

    `WRITE_ICS` : Forward advancement with saving of forward data required to
    restart and advance the forward.

    `READ_ADJ_DEPS` : Loading of adjoint dependency data.

    `READ_ICS` : Loading of forward restart data.
    """

    NONE = 0
    FORWARD = 1
    FORWARD_REVERSE = 2
    WRITE_ADJ_DEPS = 3
    WRITE_ICS = 4
    READ_ADJ_DEPS = 5
    READ_ICS = 6


class CheckpointAction:
    """Checkpointing action base class.

    Attributes
    ----------
    * args : Any
        Action parameters.
    """

    def __init__(self, *args):
        self.args = args

    def __repr__(self):

        strargs = tuple("sys.maxsize" if arg == sys.maxsize else repr(arg)
                        for arg in self.args)
        return f"{type(self).__name__}({', '.join(strargs)})"

    def __eq__(self, other):
        return isinstance(self, other) and self.args == other.args


class Forward(CheckpointAction):
    """Forward advancement action. Indicates which data should be stored, and
    the storage type.

    Attributes
    ----------
    n0 : int
        The forward should advance from the start of this step.
    n1 : int
        The forward should advance to the start of this step.
    write_ics : bool
        Whether to store forward restart data.
    write_adj_deps : bool
        Whether to store forward data required by the adjoint.
    storage : StorageType
        The storage type for the data.

    Example
    -------
    .. code-block ::

        Forward(10, 25, True, False, StorageType.RAM)

    This action is read as:

    - Advance the forward from the start of step 10 to the start of the
        step 25 (i.e over steps 10 to 24 inclusive).

    - Store forward data (`write_ics` is `True`) required to initialize the
        forward at the start of step 10 and to advance to the start of step
        25.

    - It is not necessary to store the forward data for the adjoint
        (`write_adj_deps` is `False`).

    - The forward data should be stored in memory (`storage` is\
    `StorageType.RAM`).

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
    """Adjoint advancement action.

    Attributes
    ----------
    n1 : int
        The adjoint should advance from the start of this step.
    n0 : int
        The adjoint should advance to the start of this step.
    clear_adj_deps : bool
        Whether to clear the forward data used by the adjoint.

    Example
    -------

    .. code-block ::

        Reverse(3, 2, True)

    This action is read as:

        - Advance the adjoint from the start of step 3 to the start of the step
          2 (i.e. over step 2).
        - Clear the forward data used by the adjoint (`clear_adj_deps` is
          `'True'`).
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
    """Copy action. Indicates copying of data from one storage type to another.

    Attributes
    ----------
    n : int
        The step with which the copied data is associated.
    from_storage : StorageType
        The storage type from which the data should be copied.
    to_storage : StorageType
        The storage type to which the data should be copied.

    Notes
    -----
    The data, once copied, remains accessible in the original storage type.

    Examples
    --------

    .. code-block ::

        Copy(10, StorageType.DISK, StorageType.RAM)

    This action is read as:

        - Copy the data associated with step 10, which is stored on
          `StorageType.DISK`, to `StorageType.RAM`.

    .. code-block ::

        Copy(10, StorageType.RAM, StorageType.WORK)

    This action is read as:

        - Copy the data associated with step 10, which is stored in
          `StorageType.RAM`, to working storage for use by the forward or
          adjoint.

    See Also
    --------
    :class:`StorageType`
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
    """Move action. Indicates moving of data from one storage type to another.

    Attributes
    ----------
    n : int
        The step with which the moved data is associated.
    from_storage : StorageType
        The storage type from which the data should be moved.
    to_storage : StorageType
        The storage type to which the data should be moved.

    Notes
    -----
    The data, once moved, is no longer accessible in the original storage type.

    Examples
    --------

    .. code-block ::

        Move(10, StorageType.DISK, StorageType.RAM)

    This action is read as:

        - Move the data associated with step 10, which is stored on
          `StorageType.DISK`, to `StorageType.RAM`.

    .. code-block ::

        Move(5, StorageType.DISK, StorageType.NONE)

    This action is read as:

        - Delete the data associated with step 5 from `StorageType.DISK`.

    See Also
    --------
    :class:`StorageType`
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
    """Indicates that the forward calculation has concluded.
    """

    def __init__(self):
        super().__init__()


class EndReverse(CheckpointAction):
    """Indicates that an adjoint calculation has concluded.
    """

    def __init__(self):
        super().__init__()


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
            if not hasattr(self, "iter"):
                self.iter = cls_iter(self)
            return self.iter

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
        """Whether the schedule has concluded.

        Notes
        -----
        Some schedules permit multiple adjoint calculation, and may never
        conclude.
        """

        raise NotImplementedError

    @abstractmethod
    def uses_storage_type(self, storage_type):
        """Return whether the schedule may use a type storage.

        Parameters
        ----------
        storage_type : StorageType
            The storage type to check.

        Returns
        -------
        bool
            Whether this schedule uses a given storage type.
        """

        raise NotImplementedError

    @property
    def n(self):
        """The current location of the forward.
        """

        return self._n

    @property
    def r(self):
        """The number of adjoint steps advanced.
        """

        return self._r

    @property
    def max_n(self):
        """The number of forward steps in the initial forward calculation.
        """

        return self._max_n

    @property
    def is_running(self):
        """Whether at least one action has been yielded.
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
