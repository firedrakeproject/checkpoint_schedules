#!/usr/bin/python

import argparse

# Default parameter values. Not all parameters are used by all algorithms.
defaults = {
    "uf": 1,  # Cost of a forward step.
    "ub": 1,  # Cost of a backward step.
    "up": 1,  # Cost of the loss function.
    "wd": 5,  # Cost of writing to disk.
    "rd": 5,  # Cost of reading from disk.
    "mx": None,  # Size of the period (defaults to the optimal).
    "one_read_disk": False,  # Disk checkpoints are only read once.
    "fast": False,  # Use the clode formula for mx.
    "concat": 0,  # Level of sequence concatenation.
    "print_table": None,  # File to which to print the results table.
}


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


def parse_arguments_Revolve():
    parser = argparse.ArgumentParser(description='Compute Revolve with l and cm')
    parser.add_argument("l", help="Size of the Adjoint Computation problem (number of foward steps)", type=int)
    parser.add_argument("cm", help="Memory Size", type=int)
    parser.add_argument("--uf", help="Cost of the forward steps (default: 1)", default=defaults["uf"], type=float, metavar="float", dest="uf")
    parser.add_argument("--ub", help="Cost of the backward steps (default: 1)", default=defaults["ub"], type=float, metavar="float", dest="ub")
    parser.add_argument("--concat", help="Level of concatenation between 0 and 2? (default: 0)", default=defaults["concat"], type=int, metavar="int", dest="concat")
    parser.add_argument("--print_table", help="Name of the file to print the table of results", default=defaults["print_table"], type=str, metavar="str", dest="print_table")
    return vars(parser.parse_args())


def parse_arguments_1D_Revolve():
    parser = argparse.ArgumentParser(description='Compute 1D-Revolve with l and cm')
    parser.add_argument("l", help="Size of the Adjoint Computation problem (number of foward steps)", type=int)
    parser.add_argument("cm", help="Memory Size", type=int)
    parser.add_argument("--uf", help="Cost of the forward steps (default: 1)", default=defaults["uf"], type=float, metavar="float", dest="uf")
    parser.add_argument("--ub", help="Cost of the backward steps (default: 1)", default=defaults["ub"], type=float, metavar="float", dest="ub")
    parser.add_argument("--wd", help="Cost of writting in the disk (default: 5)", default=defaults["wd"], type=float, metavar="float", dest="wd")
    parser.add_argument("--rd", help="Cost of reading in the disk (default: 5)", default=defaults["rd"], type=float, metavar="float", dest="rd")
    parser.add_argument("--one_read_disk", help="Option to force one read maximum for disk checkpoints", action='store_true')
    parser.add_argument("--concat", help="Level of concatenation between 0 and 3? (default: 0)", default=defaults["concat"], type=int, metavar="int", dest="concat")
    parser.add_argument("--print_table", help="Name of the file to print the table of results", default=defaults["print_table"], type=str, metavar="str", dest="print_table")
    return vars(parser.parse_args())


def parse_arguments_Disk_Revolve():
    parser = argparse.ArgumentParser(description='Compute Disk-Revolve with l and cm')
    parser.add_argument("l", help="Size of the Adjoint Computation problem (number of foward steps)", type=int)
    parser.add_argument("cm", help="Memory Size", type=int)
    parser.add_argument("--uf", help="Cost of the forward steps (default: 1)", default=defaults["uf"], type=float, metavar="float", dest="uf")
    parser.add_argument("--ub", help="Cost of the backward steps (default: 1)", default=defaults["ub"], type=float, metavar="float", dest="ub")
    parser.add_argument("--wd", help="Cost of writting in the disk (default: 5)", default=defaults["wd"], type=float, metavar="float", dest="wd")
    parser.add_argument("--rd", help="Cost of reading in the disk (default: 5)", default=defaults["rd"], type=float, metavar="float", dest="rd")
    parser.add_argument("--one_read_disk", help="Option to force one read maximum for disk checkpoints", action='store_true')
    parser.add_argument("--concat", help="Level of concatenation between 0 and 3? (default: 0)", default=defaults["concat"], type=int, metavar="int", dest="concat")
    parser.add_argument("--print_table", help="Name of the file to print the table of results", default=defaults["one_read_disk"], type=str, metavar="str", dest="print_table")
    return vars(parser.parse_args())


def parse_arguments_Periodic_Disk_Revolve():
    parser = argparse.ArgumentParser(description='Compute Disk-Revolve with l and cm')
    parser.add_argument("l", help="Size of the Adjoint Computation problem (number of foward steps)", type=int)
    parser.add_argument("cm", help="Memory Size", type=int)
    parser.add_argument("--uf", help="Cost of the forward steps (default: 1)", default=defaults["uf"], type=float, metavar="float", dest="uf")
    parser.add_argument("--ub", help="Cost of the backward steps (default: 1)", default=defaults["ub"], type=float, metavar="float", dest="ub")
    parser.add_argument("--wd", help="Cost of writting in the disk (default: 5)", default=defaults["wd"], type=float, metavar="float", dest="wd")
    parser.add_argument("--rd", help="Cost of reading in the disk (default: 5)", default=defaults["rd"], type=float, metavar="float", dest="rd")
    parser.add_argument("--mx", help="Size of the period (default: the optimal one)", default=defaults["mx"], type=int, metavar="int", dest="mx")
    parser.add_argument("--one_read_disk", help="Option to force one read maximum for disk checkpoints", action='store_true')
    parser.add_argument("--fast", help="Option to use the clode formula for mx (faster)", action='store_true')
    parser.add_argument("--concat", help="Level of concatenation between 0 and 3? (default: 0)", default=defaults["concat"], type=int, metavar="int", dest="concat")
    parser.add_argument("--print_table", help="Name of the file to print the table of results", default=defaults["print_table"], type=str, metavar="str", dest="print_table")
    return vars(parser.parse_args())


def parse_arguments_HRevolve():
    parser = argparse.ArgumentParser(description='Compute HRevolve with l and the architecture described in file_name')
    parser.add_argument("l", help="Size of the Adjoint Computation problem (number of foward steps)", type=int)
    parser.add_argument("file_name", help="File describing the architecture", type=str)
    parser.add_argument("--uf", help="Cost of the forward steps (default: 1)", default=defaults["uf"], type=float, metavar="float", dest="uf")
    parser.add_argument("--ub", help="Cost of the backward steps (default: 1)", default=defaults["ub"], type=float, metavar="float", dest="ub")
    parser.add_argument("--concat", help="Level of concatenation between 0 and K? (default: 0)", default=defaults["concat"], type=int, metavar="int", dest="concat")
    return vars(parser.parse_args())
