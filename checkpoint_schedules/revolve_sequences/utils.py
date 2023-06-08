#!/usr/bin/python

class Architecture:
    def __init__(self, file_name):
        self.file_name = file_name
        try:
            self.file = open(file_name, "r")
        except FileNotFoundError:
            raise FileNotFoundError("The file "+file_name+" describing the architecture is not found.")
        self.sizes = []
        self.wd = []
        self.rd = []
        self.nblevels = -1
        self.read_file()
        if (len(self.sizes) != self.nblevels) or (len(self.wd) != self.nblevels) or (len(self.rd) != self.nblevels):
            raise ImportError("The level in the architecture file does not correspond to the number of line.")
        if (sorted(self.wd) != self.wd) or (sorted(self.rd) != self.rd):
            print("WARNING!!! This code is optimal only if the costs of writing and reading of the architecture are in the increasing order for the levels.")

    def __del__(self):
        try:
            self.file.close()
        except FileNotFoundError:
            pass
        except AttributeError:
            pass

    def __repr__(self):
        l = []
        for i in range(self.nblevels):
            l.append((self.sizes[i], self.wd[i], self.rd[i]))
        return l.__repr__()

    def read_file(self):
        for line in self.file:
            if line[0] == "#":
                continue
            line_list = [x for x in line.split()]
            if len(line_list) > 1 and self.nblevels < 0:
                raise SyntaxError("The first line of the architecture file should be a single integer.")
            if self.nblevels < 0:
                try:
                    self.nblevels = int(line_list[0])
                except:  # NOQA E722
                    raise SyntaxError("The first line of the architecture file should be an integer.")
                continue
            if len(line_list) != 3:
                raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
            try:
                self.sizes.append(int(line_list[0]))
            except:  # NOQA E722
                raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
            try:
                self.wd.append(int(line_list[1]))
            except:  # NOQA E722
                try:
                    self.wd.append(float(line_list[1]))
                except:  # NOQA E722
                    raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
            try:
                self.rd.append(int(line_list[2]))
            except:  # NOQA E722
                try:
                    self.rd.append(float(line_list[2]))
                except:  # NOQA E722
                    raise SyntaxError("Every line of the architecture file should be a triplet (c: integer, wd: float, rd: float).")
        self.file.seek(0)



def revolver_parameters(wd, rd):
    """Set default revolver parameters.
    
    Parameters
    ----------
    wd : float|tuple
        ....
    rd : float|tuple
        ....

    Notes
    -----

    Returns:
        dict: Revolver parameters.
    """

    params = {
        "uf": 1,  # Cost of a forward step.
        "ub": 1,  # Cost of a backward step.
        "up": 1,  # Cost of the loss function.
        "wd": wd,  # Cost of writing to disk.
        "rd": rd,  # Cost of reading from disk.
        "mx": 2,  # Size of the period (defaults to the optimal).
        "one_read_disk": False,  # Disk checkpoints are only read once.
        "fast": False,  # Use the clode formula for mx.
        "concat": 0,  # Level of sequence concatenation.
        "print_table": "None",  # File to which to print the results table.
    }
    return params
