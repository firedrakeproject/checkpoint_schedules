# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr),
# Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains the basic functions used in the H-ReVolve algorithm."""
import math

official_names = {
    "Forward": "F",
    "Backward": "B",
    "Checkpoint": "C",
    "Read_disk": "RD",
    "Write_disk": "WD",
    "Read_memory": "RM",
    "Write_memory": "WM",
    "Discard_disk": "DD",
    "Discard_memory": "DM",
    "Read": "R",
    "Write": "W",
    "Discard": "D",
    "Discard_Forward": "DF",
    "Forward_branch": "F",
    "Backward_branch": "B",
    "Turn": "T",
    "Write_Forward": "WF",
    "Write_Forward_memory": "WFM",
    "Discard_branch": "DB",
    "Discard_Forward_branch": "DFB",
    "Checkpoint_branch": "C",
    "Discard_Forward_disk": "DFD",
    "Discard_Forward_memory": "DFM",
}


def beta(x, y):
    """This function auxiliate in the optimal makespan computation.

    Parameters
    ----------
    x : float
        The number of slots available in memory.
    y : int
        It is the unic inter satisfying the following inequality

        .. math::

            \\frac{(x+y)!}{x!y!} \\leq l \\leq \\frac{(x+y+1)!}{x!(y+1)!}

    Returns
    -------
    int
        This function returns the value that contributes for the optimal
        makespan computation.
    """
    if y < 0:
        return 0
    return math.factorial(x+y) / (math.factorial(x) * math.factorial(y))


def argmin(list):
    """Provide the index of the minimum value of the memory list.
    It is used to compute operation index in the H-ReVolve
    schedule for K = 0 (level 0).

    Parameters
    ----------
    l : list
        The list of memory.

    Returns
    -------
    int
        Index of the minimum value in the memory list.
    """
    index = 0
    m = list[0]
    for i, _ in enumerate(list):
        if list[i] <= m:
            index = i
            m = list[i]
    return 1 + index


def from_list_to_string(list):
    """Convert a ist to a string.

    Parameters
    ----------
    l : list
        The list to be converted.

    Returns
    -------
    str
        The string representation of the list.
    """
    s = ""
    for x in list[:-1]:
        s += str(x) + ", "
    s = s[:-2] + "; " + str(list[-1])
    return "(" + s + ")"


class Operation:
    """This class represents the operations given by the checkpointing
    strategies.

    Attributes
    ----------
    type : str
        The type of operation.
    index : int or list of int
        The index of the operation.
    params : dict
        It is dictionary of parameters.

    Notes
    -----
    If it is an H-Revolve schedule, the index is a list of two integers.
    The first integer represents the storage hierarchical level, and the
    second integer represents the time step. Otherwise, the index is an integer
    representing the time step.
    The possible types of operations are listed in `official_names`.
    The dictionary of parameters is defined in
    :py:func:`utils.revolver_parameters`.

    See Also
    --------
    :func:`utils.revolver_parameters`, `official_names`

    """
    def __init__(self, operation_type, operation_index, params):
        if operation_type not in official_names:
            raise ValueError("Unreconized operation name: " + operation_type)
        self.type = operation_type
        self.index = operation_index
        self.params = params

    def __repr__(self):
        if self.index is None:
            return official_names[self.type]
        if isinstance(self.index, int):
            # type(self.index) is int:
            return official_names[self.type] + "_" + str(self.index)
        elif isinstance(self.index, list):
            # type(self.index) is list:
            if self.type == "Forward" or self.type == "Backward":
                return official_names[self.type] + "_" + str(self.index[0]) + \
                    "->" + str(self.index[1])
            elif self.type == "Forward_branch":
                return official_names[self.type] + "^" + str(self.index[0]) + \
                    "_" + str(self.index[1]) + "->" + str(self.index[2])
            else:
                return official_names[self.type] + "^" + str(self.index[0]) + \
                    "_" + str(self.index[1])

    def cost(self):
        """Cost of the operations.

        Returns
        -------
        float
            The cost.
        """
        if self.type == "Forward":
            return (self.index[1] - self.index[0]) * self.params["uf"]
        if self.type == "Backward":
            return self.params["ub"]
        if self.type == "Checkpoint":
            return 0
        if self.type == "Read_disk":
            return self.params["rd"]
        if self.type == "Write_disk":
            return self.params["wd"]
        if self.type == "Read_memory":
            return 0
        if self.type == "Write_memory":
            return 0
        if self.type == "Write_Forward_memory":
            return 0
        if self.type == "Discard_disk":
            return 0
        if self.type == "Discard_memory":
            return 0
        if self.type == "Discard_Forward_disk":
            return 0
        if self.type == "Discard_Forward_memory":
            return 0
        if self.type == "Read":
            return self.params["rd"][self.index[0]]
        if self.type == "Write":
            return self.params["wd"][self.index[0]]
        if self.type == "Write_Forward":
            return self.params["wd"][self.index[0]]
        if self.type == "Discard":
            return 0
        if self.type == "Discard_Forward":
            return 0
        if self.type == "Forward_branch":
            return (self.index[2] - self.index[1]) * self.params["uf"]
        if self.type == "Backward_branch":
            return self.params["cbwd"]
        if self.type == "Turn":
            return self.params["up"]
        if self.type == "Discard_branch":
            return 0
        if self.type == "Discard_Forward_branch":
            return 0
        if self.type == "Checkpoint_branch":
            return 0
        raise ValueError("Unknown cost for operation type " + self.type)

    def shift(self, size, branch=-1):
        """Shift the index of the operation.

        Parameters
        ----------
        size : int
            The index size to shift.
        branch : int, optional
            The operation branch.

        Notes
        -----
        The shift is applied to the index that represents the time step of the
        operation.
        """
        if isinstance(self.index, int):
            self.index += size
        elif isinstance(self.index, list):
            if self.type == "Forward" or self.type == "Backward":
                self.index[0] += size
                self.index[1] += size
            elif self.type == "Forwards_multi":
                self.index[1] += size
                self.index[2] += size
            elif self.type in ["Forward_branch", "Discard_branch",
                               "Discard_Forward_branch",
                               "Checkpoint_branch", "Backward_branch"]:
                if self.index[0] == branch:
                    for i in range(1, len(self.index)):
                        self.index[i] += size
            else:
                self.index[1] += size


class Function:
    """This class creates the H-Revolve functions.

    Attributes
    ----------
    name : str
        Name of the function.
    l : int
        The storage type of the function.
    index : int or list
        The index of the function.  If is a H-ReVolve schedule, the index is a
        list of two integers. The first integer is the storage type
        (either `'RAM'` or `'disk'`),
        and the second integer is the the time step. Otherwise, the index is
        an integer representing the time step.
    """
    def __init__(self, name, list, index):
        self.name = name
        self.list = list
        self.index = index

    def __repr__(self):
        if self.name == "HRevolve" or self.name == "hrevolve_aux":
            return self.name + "_" + str(self.index[0]) + "(" + str(self.list)\
                + ", " + str(self.index[1]) + ")"
        else:
            return self.name + "(" + str(self.list) + ", " + str(self.index) + ")"   # noqa: E501


class Sequence:
    """This class creates the Revolve sequences.

    Attributes
    ----------
    sequence : list
        List create to store the :class:`Operation` and :class:`Sequence`.
    function : Function
        Description the function (name and parameters).
    levels : int
        The number of levels in the hierarchical storage.
    concat : int
        Give the output format for the returned sequence.
    makespan : int
        Represent the total execution time of a sequence.
    storage : list
        List of list of checkpoints in hierarchical storage
    memory : list
        List of memory checkpoints.
    disk : list
        List of disk checkpoints.
    type : str
        Type of the sequence.

    Notes
    -----
    The possible types are listed in :attr:`official_names`.

    """
    def __init__(self, function, levels=None, concat=0):
        self.sequence = []
        self.function = function
        self.levels = levels
        self.concat = concat
        self.makespan = 0
        if (self.function.name == "HRevolve" or self.function.name == "hrevolve_aux"):  # noqa: E501
            self.storage = [[] for _ in range(self.levels)]
        else:
            self.memory = []
            self.disk = []
        self.type = "Function"

    def __repr__(self):
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":  # noqa: E501
            return self.concat_sequence_hierarchic(self.concat).__repr__()
        else:
            if self.concat == 3:
                return from_list_to_string(self.canonical())
            else:
                return self.concat_sequence(self.concat).__repr__()

    def __iter__(self):
        return iter(self.concat_sequence(self.concat))

    def canonical(self):
        """Return the canonical sequence.
        """
        if self.function.name == "Disk-Revolve":
            concat = 2
        if self.function.name == "1D-Revolve" or self.function.name == "Revolve":  # noqa: E501
            concat = 1
        list = [x.l + 1 for x in self.concat_sequence(concat=concat)
                if x.__class__.__name__ == "Function"]
        list.reverse()
        return list

    def concat_sequence(self, concat):
        """Concatenate the sequence.

        Parameters
        ----------
        concat : int
            Give the output format for the returned sequence.

        Returns
        -------
        list
            The concatenated sequence.
        """
        list = []
        for x in self.sequence:
            if x.__class__.__name__ == "Operation":
                list.append(x)
            elif x.__class__.__name__ == "Sequence":
                if self.concat == 0:
                    list += x.concat_sequence(concat=concat)
                elif concat == 1:
                    if x.function.name == "Revolve":
                        list.append(x.function)
                    else:
                        list += x.concat_sequence(concat=concat)
                elif concat == 2:
                    if x.function.name == "Revolve" or x.function.name == "1D-Revolve":  # noqa: E501
                        list.append(x.function)
                    else:
                        list += x.concat_sequence(concat=concat)
                else:
                    raise ValueError("Unknown concat value: " + str(concat))
            else:
                raise ValueError("Unknown class name: " + x.__class__.__name__)
        return list

    def concat_sequence_hierarchic(self, concat):
        """Concatenate the sequence in hierarchical storage.

        Parameters
        ----------
        concat : int
            Give the output format for the returned sequence.

        Returns
        -------
        list
            The concatenated sequence.
        """
        list = []
        for x in self.sequence:
            if x.__class__.__name__ == "Operation":
                list.append(x)
            elif x.__class__.__name__ == "Sequence":
                if concat == 0:
                    list += x.concat_sequence_hierarchic(concat=concat)
                elif x.function.name == "HRevolve" and x.function.index[0] <= concat-1:  # noqa: E501
                    list.append(x.function)
                else:
                    list += x.concat_sequence_hierarchic(concat=concat)
            else:
                raise ValueError("Unknown class name: " + x.__class__.__name__)
        return list

    def insert(self, operation):
        """Insert an operation in the sequence.

        Parameters
        ----------
        operation : Operation
            The operation to insert.
        """
        self.sequence.append(operation)
        self.makespan += operation.cost()
        if operation.type == "Write_memory":
            self.memory.append(operation.index)
        if operation.type == "Write_Forward_memory":
            self.memory.append(operation.index)
        if operation.type == "Write_disk":
            self.disk.append(operation.index)
        if operation.type == "Checkpoint":
            self.memory.append(operation.index)
        if operation.type == "Write":
            self.storage[operation.index[0]].append(operation.index[1])
        if operation.type == "Checkpoint_branch":
            self.memory.append((operation.index[0], operation.index[1]))

    def remove(self, operation_index):
        """Remove an operation in the sequence.

        Parameters
        ----------
        operation_index : int
            The index of the operation to remove.
        """
        self.makespan -= self.sequence[operation_index].cost()
        if self.sequence[operation_index].type == "Write_memory":
            self.memory.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write_Forward_memory":
            self.memory.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write_disk":
            self.disk.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write":
            self.storage[self.sequence[operation_index].index[0]].remove(
                self.sequence[operation_index].index[1])
        if self.sequence[operation_index].type == "Checkpoint":
            self.memory.remove(self.sequence[operation_index].index)
        del self.sequence[operation_index]

    def insert_sequence(self, sequence):
        """Insert a sequence into the current sequence.

        Parameters
        ----------
        sequence : Sequence
            The sequence to insert.
        """
        self.sequence.append(sequence)
        self.makespan += sequence.makespan
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":  # noqa: E501
            for i in range(len(self.storage)):
                self.storage[i] += sequence.storage[i]
        else:
            self.memory += sequence.memory
            self.disk += sequence.disk

    def shift(self, size, branch=-1):
        """Shift the index of the operation within this sequence.

        Parameters
        ----------
        size : int
            The size of the shift.
        branch : int, optional
            The operation branch.

        Returns
        -------
        Sequence
            The updated sequence with the operation indexes shifted.
        """
        for x in self.sequence:
            x.shift(size, branch=branch)
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":  # noqa: E501
            for i in range(len(self.storage)):
                self.storage[i] = [x + size for x in self.storage[i]]
        else:
            self.memory = [x + size if type(x) is int else (x[0], x[1] + size)
                           if x[0] == branch else (x[0], x[1])
                           for x in self.memory]
            self.disk = [x + size if type(x) is int else (x[0], x[1] + size)
                         if x[0] == branch else (x[0], x[1])
                         for x in self.disk]
        return self

    def remove_useless_wm(self, K=-1):
        """Remove useless write in memory operations from the sequence.

        Parameters
        ----------
        K : int, optional
            Index of the write storage level.

        Returns
        -------
        Sequence
            The updated sequence without useless write in-memory operations.
        """
        if len(self.sequence) > 0:
            if self.sequence[0].type == "Write_memory" or self.sequence[0].type == "Checkpoint":  # noqa: E501
                self.remove(0)
                return self
        if len(self.sequence) > 0:
            if self.sequence[0].type == "Write":
                if self.sequence[0].index[0] == K:
                    self.remove(0)
                    return self
        return self

    def remove_last_discard(self):
        """Remove the last discard operation.
        """
        if self.sequence[-1].type == "Function":
            self.sequence[-1].remove_last_discard()
        if self.sequence[-1].type in ["Discard_memory", "Discard_disk",
                                      "Discard", "Discard_branch"]:
            self.remove(-1)

    def first_operation(self):
        """Get the first operation of the sequence.

        Returns
        -------
        Operation
            The first operation of the sequence.
        """
        if self.sequence[0].type == "Function":
            return self.sequence[0].first_operation()
        else:
            return self.sequence[0]

    def next_operation(self, i):
        """Get the next operation of the sequence.

        Parameters
        ----------
        i : int
            The index of the current operation.

        Returns
        -------
        Operation
            The next operation of the sequence.
        """
        if self.sequence[i+1].type == "Function":
            return self.sequence[i+1].first_operation()
        else:
            return self.sequence[i+1]

    def convert_old_to_branch(self, index):
        """Convert an old operation to a branch operation.

        Parameters
        ----------
        index : int
            The index of the branch.

        Returns
        -------
        Sequence
            The sequence with the converted operation.
        """
        for (i, x) in enumerate(self.memory):
            if type(x) is int:
                self.memory[i] = (index, x)
        to_remove = []
        for (i, op) in enumerate(self.sequence):
            if op.type == "Function":
                self.sequence[i] = self.sequence[i].convert_old_to_branch(index)  # noqa: E501
            elif op.type == "Forward":
                op.type = "Forward_branch"
                op.index = [index] + op.index
            elif op.type == "Backward":
                op.type = "Backward_branch"
                op.index = [index, op.index]
            elif op.type == "Read":
                if self.next_operation(i).type == "Backward":
                    to_remove.append(i)
                else:
                    op.type = "Checkpoint_branch"
                    op.index = [index, op.index]
            elif op.type == "Write":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type == "Discard":
                to_remove.append(i)
            elif op.type == "Read_memory":
                if self.next_operation(i).type == "Backward":
                    to_remove.append(i)
                else:
                    op.type = "Checkpoint_branch"
                    op.index = [index, op.index]
            elif op.type == "Write_memory":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type == "Discard_memory":
                to_remove.append(i)
            elif op.type in ["Read_disk", "Write_disk", "Discard_disk"]:
                raise ValueError("Cannot use convert_old_to_branch on \
                                 sequences from two-memory architecture")
            else:
                raise ValueError("Unknown data type %s in \
                                 convert_old_to_branch" + op.type)
        for (i, index) in enumerate(to_remove):
            self.remove(index-i)
        return self

    def convert_new_to_branch(self, index):
        """Convert a new operation to a branch operation.

        Parameters
        ----------
        index : int
            The index of the branch.

        Returns
        -------
        Sequence
            The sequence with the converted operation.
        """
        for (i, x) in enumerate(self.memory):
            if isinstance(x, int):
                self.memory[i] = (index, x)
        to_remove = []
        for (i, op) in enumerate(self.sequence):
            if op.type == "Function":
                self.sequence[i] = self.sequence[i].convert_new_to_branch(index)  # noqa: E501
            elif op.type == "Forward":
                op.type = "Forward_branch"
                op.index = [index] + op.index
            elif op.type == "Backward":
                op.type = "Backward_branch"
                op.index = [index, op.index]
            elif op.type == "Checkpoint":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type in ["Forward_branch", "Turn", "Discard_branch",
                             "Checkpoint", "Backward_branch"]:
                continue
            elif op.type in ["Read_disk", "Write_disk", "Discard_disk"]:
                raise ValueError("Cannot use convert_new_to_branch on \
                                  sequences from two-memory architecture")
            else:
                raise ValueError("Unknown data type: " + op.type)
        for (i, index) in enumerate(to_remove):
            self.remove(index-i)
        return self


class Table:
    """This class creates a Table.

    Attributes
    ----------
    content : list
        The content of the table.
    size : int
        The size of the table.
    """
    def __init__(self, n=0, x=float("inf")):
        self.content = [x for _ in range(n)]
        self.size = n
        self.print_table = 0

    def set_to_print(self, file_name):
        """Print the table to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to print to.
        """
        self.print_table = 1
        self.file = open(file_name, "w")
        self.file.write("#l\tvalue\n")

    def append(self, x):
        """Appends an element to the table content.

        Parameters
        ----------
        x : int
            The element to append to the table content.

        Notes
        -----
        The value of 'x' represents a function of the forward and backward
        operation costs.

        """
        self.content.append(x)
        self.size += 1
        if self.print_table:
            self.file.write("%d\t%d\n" % (self.size-1, x))
            if self.size % 10 == 0:
                self.file.flush()

    def remove(self, x):
        """Remove an element from the table.

        Parameters
        ----------
        x : ..
            The element to remove.
        """
        self.content.remove(x)
        self.size -= 1

    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError("Index out of range. Table length: " +
                             str(len(self)) + "," + "Index: " + str(i))
        return self.content[i]

    def __repr__(self):
        return self.content.__repr__()

    def __del__(self):
        if self.print_table == 1:
            self.file.flush()
            self.file.close()

    def __len__(self):
        return self.content.__len__()
