# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr),
# Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains the functions used to compute the 1D revolver
sequences.
"""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, Table,
                              argmin)
from .revolve import revolve, get_opt_0_table


def get_opt_1d_table(lmax, cm, ub, uf, rd, one_read_disk, print_table=None,
                     opt_0=None):
    """Compute the execution time of the 1D revolver algorithm.

    Parameters
    ----------
    lmax : int
        The number of forward steps to use in the AC (Adjoint Computation)
        graph.
    cm : int
        Number of memory slots.
    ub : float
        The cost of advancing the adjoint over one step.
    uf : float
        The cost of advancing the forward over one step.
    rd : float
        Cost of reading the checkpoint data from disk.
    one_read_disk : bool
        Disk checkpoints are only read once.
    print_table : str, optional
        File to which to print the results table.
    opt_0 : list, optional
        Optimal execution time for a memory revolver algorithm.

    Notes
    -----
    This computation uses a dynamic program.
    One considers that 'x_0' is already stored on the disk.
    The schedule building is the execution time of an optimal solution.
    In this case, the optimal solution is given as a function of the `cm`,
    `lmax`, and `rd`. Additional details about the execution time is avaiable
    in the paper [1], at the Theorem 3.15.

    [1] Aupy, G.,  Herrmann, Ju. and Hovland, P. and Robert, Y.
    "Optimal multistage algorithm for adjoint computation". SIAM Journal on
    Scientific Computing, 38(3),
    C232-C255, (2016). DOI: 10.1145/347837.347846

    Returns
    -------
    list
        Optimal execution time of the 1D revolver algorithm.
    """
    if opt_0 is None:
        opt_0 = get_opt_0_table(lmax, cm, uf, ub, print_table)
    opt_1d = Table()
    if __name__ == '__main__' and print_table:
        opt_1d.set_to_print(print_table)
    opt_1d.append(ub)
    # Opt_1d[1] for cm
    if cm == 0:
        opt_1d.append(uf + 2 * ub + rd)
    else:
        opt_1d.append(uf + 2 * ub)
    # Opt_1d[2...lmax] for cm
    for l in range(2, lmax + 1):  # noqa: E741
        if one_read_disk:
            m = min([j * uf + opt_0[cm][l - j] + rd + opt_0[cm][j-1]
                     for j in range(1, l)])
            opt_1d.append(min(opt_0[cm][l], m))
        else:
            m = min([j * uf + opt_0[cm][l - j] + rd + opt_1d[j-1]
                     for j in range(1, l)])
            opt_1d.append(min(opt_0[cm][l], m))
    return opt_1d


def revolve_1d(l, cm, opt_0=None, opt_1d=None, **params):  # noqa: E741
    """1D revolve algorithm.

    Parameters
    ----------
    l : int
        The number of forward step to execute in the AC graph.
    cm : int
        The maximum number of checkpoints to store in memory.
    opt_0 : list, optional
        Optimal execution time for a memory revolver algorithm.
    opt_1d : lis, optional
        Optimal execution time for a 1D revolver algorithm.

    Notes
    -----
    This algorithm is a subroutine of Disk-Revolve.
    1D revolve uses only on checkpoint slot in the second level of storage.

    Returns
    -------
    Sequence
        1D revolve schedule.
    """
    parameters = dict(params)
    rd = parameters["rd"]
    uf = parameters["uf"]
    wd = parameters["wd"]
    ub = parameters["ub"]
    one_read_disk = parameters["one_read_disk"]
    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, **parameters)
    if opt_1d is None:
        opt_1d = get_opt_1d_table(l, cm, opt_0=opt_0, **parameters)
    sequence = Sequence(Function("1D-Revolve", l, cm),
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:  # noqa: E741
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory", 1))
        return sequence
    if l == 1:  # noqa: E741
        if cm == 0:
            sequence.insert(operation("Forward", [0, 1]))
            sequence.insert(operation("Write_Forward_memory", 2))
            sequence.insert(operation("Forward", [1, 2]))
            sequence.insert(operation("Backward", [2, 1]))
            sequence.insert(operation("Discard_Forward_memory", 2))
            sequence.insert(operation("Read_disk", 0))
            sequence.insert(operation("Write_Forward_memory", 1))
            sequence.insert(operation("Forward", [0, 1]))
            sequence.insert(operation("Backward", [1, 0]))
            sequence.insert(operation("Discard_Forward_memory", 1))
            return sequence
        else:
            sequence.insert(operation("Write_memory", 0))
            sequence.insert(operation("Forward", [0, 1]))
            sequence.insert(operation("Write_Forward_memory", 2))
            sequence.insert(operation("Forward", [1, 2]))
            sequence.insert(operation("Backward", [2, 1]))
            sequence.insert(operation("Discard_Forward_memory", 2))
            sequence.insert(operation("Read_memory", 0))
            sequence.insert(operation("Write_Forward_memory", 1))
            sequence.insert(operation("Forward", [0, 1]))
            sequence.insert(operation("Backward", [1, 0]))
            sequence.insert(operation("Discard_Forward_memory", 1))
            sequence.insert(operation("Discard_memory", 0))
            return sequence
    if one_read_disk:
        list_mem = [j * uf + opt_0[cm][l - j] + rd + opt_0[cm][j-1]
                    for j in range(1, l)]
    else:
        list_mem = [j * uf + opt_0[cm][l - j] + rd + opt_1d[j-1]
                    for j in range(1, l)]
    if min(list_mem) < opt_0[cm][l]:
        jmin = argmin(list_mem)
        sequence.insert(operation("Forward", [0, jmin]))
        sequence.insert_sequence(
            revolve(l - jmin, cm, opt_0=opt_0, **parameters).shift(jmin)
        )
        sequence.insert(operation("Read_disk", 0))
        if one_read_disk:
            sequence.insert_sequence(
                revolve(jmin - 1, cm, opt_0=opt_0, **parameters)
            )
        else:
            sequence.insert_sequence(
                revolve_1d(jmin - 1, cm, opt_0=opt_0, opt_1d=opt_1d,
                           **parameters)
            )
        return sequence
    else:
        sequence.insert_sequence(revolve(l, cm, rd, wd, uf, ub, opt_0=opt_0))
        return sequence
