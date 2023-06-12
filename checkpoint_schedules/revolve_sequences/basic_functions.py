#!/usr/bin/python

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
    """_summary_

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if y < 0:
        return 0
    return math.factorial(x+y) / (math.factorial(x) * math.factorial(y))


def argmin(l):
    """_summary_

    Parameters
    ----------
    l : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Return the last argmin (1-based)
    index = 0
    m = l[0]
    for i in range(len(l)):
        if l[i] <= m:
            index = i
            m = l[i]
    return 1 + index


def argmin0(l):
    """_summary_

    Parameters
    ----------
    l : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Return the first argmin (0-based)
    index = 0
    m = l[0]
    for i in range(len(l)):
        if l[i] < m:
            index = i
            m = l[i]
    return index


def argmin_list(l):
    """_summary_

    Parameters
    ----------
    l : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    index_list = [0]
    m = l[0]
    for i in range(len(l)):
        if l[i] < m:
            index_list = [i+1]
            m = l[i]
        if l[i] == m:
            index_list.append(i+1)
    return index_list


def from_list_to_string(l):
    """_summary_

    Parameters
    ----------
    l : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    s = ""
    for x in l[:-1]:
        s += str(x) + ", "
    s = s[:-2] + "; " + str(l[-1])
    return "(" + s + ")"


class Operation:
    """_summary_
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
                return official_names[self.type] + "_" + str(self.index[0]) + "->" + str(self.index[1])
            elif self.type == "Forward_branch":
                return official_names[self.type] + "^" + str(self.index[0]) + "_" + str(self.index[1]) + "->" + str(self.index[2])
            else:
                return official_names[self.type] + "^" + str(self.index[0]) + "_" + str(self.index[1])

    def cost(self):
        """_summary_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
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
        """_summary_

        Parameters
        ----------
        size : _type_
            _description_
        branch : int, optional
            _description_, by default -1
        """
        if type(self.index) is int:
            self.index += size
        elif type(self.index) is list:
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
    """_summary_
    """
    def __init__(self, name, l, index):
        self.name = name
        self.l = l
        self.index = index

    def __repr__(self):
        if self.name == "HRevolve" or self.name == "hrevolve_aux":
            return self.name + "_" + str(self.index[0]) + "(" + str(self.l) + ", " + str(self.index[1]) + ")"
        else:
            return self.name + "(" + str(self.l) + ", " + str(self.index) + ")"


class Sequence:
    """_summary_
    """
    def __init__(self, function, levels=None, concat=0):
        self.sequence = []  # List of Operation and Sequence
        self.function = function  # Description the function (name and parameters)
        self.levels = levels
        self.concat = concat
        self.makespan = 0  # Makespan to be updated
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":
            self.storage = [[] for _ in range(self.levels)]  # List of list of checkpoints in hierarchical storage
        else:
            self.memory = []  # List of memory checkpoints
            self.disk = []  # List of disk checkpoints
        self.type = "Function"

    def __repr__(self):
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":
            return self.concat_sequence_hierarchic(self.concat).__repr__()
        else:
            if self.concat == 3:
                return from_list_to_string(self.canonical())
            else:
                return self.concat_sequence(self.concat).__repr__()

    def __iter__(self):
        return iter(self.concat_sequence(self.concat))

    def canonical(self):
        if self.function.name == "Disk-Revolve":
            concat = 2
        if self.function.name == "1D-Revolve" or self.function.name == "Revolve":
            concat = 1
        l = [x.l + 1 for x in self.concat_sequence(concat=concat) if x.__class__.__name__ == "Function"]
        l.reverse()
        return l

    def concat_sequence(self, concat):
        """_summary_

        Parameters
        ----------
        concat : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        l = []
        for x in self.sequence:
            if x.__class__.__name__ == "Operation":
                l.append(x)
            elif x.__class__.__name__ == "Sequence":
                if self.concat == 0:
                    l += x.concat_sequence(concat=concat)
                elif concat == 1:
                    if x.function.name == "Revolve":
                        l.append(x.function)
                    else:
                        l += x.concat_sequence(concat=concat)
                elif concat == 2:
                    if x.function.name == "Revolve" or x.function.name == "1D-Revolve":
                        l.append(x.function)
                    else:
                        l += x.concat_sequence(concat=concat)
                else:
                    raise ValueError("Unknown concat value: " + str(concat))
            else:
                raise ValueError("Unknown class name: " + x.__class__.__name__)
        return l

    def concat_sequence_hierarchic(self, concat):
        """_summary_

        Parameters
        ----------
        concat : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        l = []
        for x in self.sequence:
            if x.__class__.__name__ == "Operation":
                l.append(x)
            elif x.__class__.__name__ == "Sequence":
                if concat == 0:
                    l += x.concat_sequence_hierarchic(concat=concat)
                elif x.function.name == "HRevolve" and x.function.index[0] <= concat-1:
                    l.append(x.function)
                else:
                    l += x.concat_sequence_hierarchic(concat=concat)
            else:
                raise ValueError("Unknown class name: " + x.__class__.__name__)
        return l

    def insert(self, operation):
        """_summary_

        Parameters
        ----------
        operation : _type_
            _description_
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
        """_summary_

        Parameters
        ----------
        operation_index : _type_
            _description_
        """
        self.makespan -= self.sequence[operation_index].cost()
        if self.sequence[operation_index].type == "Write_memory":
            self.memory.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write_Forward_memory":
            self.memory.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write_disk":
            self.disk.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write":
            self.storage[self.sequence[operation_index].index[0]].remove(self.sequence[operation_index].index[1])
        if self.sequence[operation_index].type == "Checkpoint":
            self.memory.remove(self.sequence[operation_index].index)
        del self.sequence[operation_index]

    def insert_sequence(self, sequence):
        """_summary_

        Parameters
        ----------
        sequence : _type_
            _description_
        """
        self.sequence.append(sequence)
        self.makespan += sequence.makespan
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":
            for i in range(len(self.storage)):
                self.storage[i] += sequence.storage[i]
        else:
            self.memory += sequence.memory
            self.disk += sequence.disk

    def shift(self, size, branch=-1):
        """_summary_

        Parameters
        ----------
        size : _type_
            _description_
        branch : int, optional
            _description_, by default -1

        Returns
        -------
        _type_
            _description_
        """
        for x in self.sequence:
            x.shift(size, branch=branch)
        if self.function.name == "HRevolve" or self.function.name == "hrevolve_aux":
            for i in range(len(self.storage)):
                self.storage[i] = [x + size for x in self.storage[i]]
        else:
            self.memory = [x + size if type(x) is int else (x[0], x[1] + size) if x[0] == branch else (x[0], x[1]) for x in self.memory]
            self.disk = [x + size if type(x) is int else (x[0], x[1] + size) if x[0] == branch else (x[0], x[1]) for x in self.disk]
        return self

    def remove_useless_wm(self, K=-1):
        """_summary_

        Parameters
        ----------
        K : int, optional
            _description_, by default -1

        Returns
        -------
        _type_
            _description_
        """
        if len(self.sequence) > 0:
            if self.sequence[0].type == "Write_memory" or self.sequence[0].type == "Checkpoint":
                self.remove(0)
                return self
        if len(self.sequence) > 0:
            if self.sequence[0].type == "Write":
                if self.sequence[0].index[0] == K:
                    self.remove(0)
                    return self
        return self

    def remove_last_discard(self):
        """_summary_
        """
        if self.sequence[-1].type == "Function":
            self.sequence[-1].remove_last_discard()
        if self.sequence[-1].type in ["Discard_memory", "Discard_disk", "Discard", "Discard_branch"]:
            self.remove(-1)

    def first_operation(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        if self.sequence[0].type == "Function":
            return self.sequence[0].first_operation()
        else:
            return self.sequence[0]

    def next_operation(self, i):
        """_summary_

        Parameters
        ----------
        i : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if self.sequence[i+1].type == "Function":
            return self.sequence[i+1].first_operation()
        else:
            return self.sequence[i+1]

    def convert_old_to_branch(self, index):
        """_summary_

        Parameters
        ----------
        index : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        for (i, x) in enumerate(self.memory):
            if type(x) is int:
                self.memory[i] = (index, x)
        to_remove = []
        for (i, op) in enumerate(self.sequence):
            if op.type == "Function":
                self.sequence[i] = self.sequence[i].convert_old_to_branch(index)
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
                ValueError("Cannot use convert_old_to_branch on sequences from two-memory architecture")
            else:
                ValueError("Unknown data type %s in convert_old_to_branch" % op.type)
        for (i, index) in enumerate(to_remove):
            self.remove(index-i)
        return self

    def convert_new_to_branch(self, index):
        """_summary_

        Parameters
        ----------
        index : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        for (i, x) in enumerate(self.memory):
            if type(x) is int:
                self.memory[i] = (index, x)
        to_remove = []
        for (i, op) in enumerate(self.sequence):
            if op.type == "Function":
                self.sequence[i] = self.sequence[i].convert_new_to_branch(index)
            elif op.type == "Forward":
                op.type = "Forward_branch"
                op.index = [index] + op.index
            elif op.type == "Backward":
                op.type = "Backward_branch"
                op.index = [index, op.index]
            elif op.type == "Checkpoint":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type in ["Forward_branch", "Turn", "Discard_branch", "Checkpoint", "Backward_branch"]:
                continue
            elif op.type in ["Read_disk", "Write_disk", "Discard_disk"]:
                ValueError("Cannot use convert_new_to_branch on sequences from two-memory architecture")
            else:
                ValueError("Unknown data type %s in convert_new_to_branch" % op.type)
        for (i, index) in enumerate(to_remove):
            self.remove(index-i)
        return self


class Table:
    """_summary_
    """
    def __init__(self, n=0, x=float("inf")):
        self.content = [x for _ in range(n)]
        self.size = n
        self.print_table = 0

    def set_to_print(self, file_name):
        """_summary_

        Parameters
        ----------
        file_name : _type_
            _description_
        """
        self.print_table = 1
        self.file = open(file_name, "w")
        self.file.write("#l\tvalue\n")

    def append(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        self.content.append(x)
        self.size += 1
        if self.print_table:
            self.file.write("%d\t%d\n" % (self.size-1, x))
            if self.size % 10 == 0:
                self.file.flush()

    def remove(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        """
        self.content.remove(x)
        self.size -= 1

    def __getitem__(self, i):
        try:
            return self.content[i]
        except IndexError:
            raise IndexError("In table of length %d, index %d does not exist." % (len(self), i))

    def __repr__(self):
        return self.content.__repr__()

    def __del__(self):
        if self.print_table == 1:
            self.file.flush()
            self.file.close()

    def __len__(self):
        return self.content.__len__()
