# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr),
# Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains the functions to compute the Disk-Revolve schedules.
"""
from functools import partial
from .basic_functions import (Operation as Op, Table, Sequence, Function,
                              argmin)
from .revolve import get_opt_0_table, revolve
from .revolve_1d import revolve_1d, get_opt_1d_table
from .utils import revolver_parameters


def get_opt_inf_table(lmax, cm, uf, ub, rd, wd, one_read_disk,
                      print_table=None, opt_0=None, opt_1d=None):
    """Compute the optimal execution time for the Disk-Revolve algorithm.

    Parameters
    ----------
    lmax : int
        The number of forward steps to use in the AC graph.
    cm : int
        Slots stored in memory.
    ub : float
        The cost of advancing the adjoint over one step.
    uf : float
        The cost of advancing the forward over one step.
    rd : float
        Cost of reading the checkpoint data from disk.
    wd : float
        Cost of writing the checkpoint data in disk.
    one_read_disk : bool
        Disk checkpoints are only read once.
    print_table : str, optional
        File to which to print the results table, by default None.
    opt_0 : list, optional
        Optimal execution time for a memory revolve algorithm.
    opt_1 : list, optional
        Optimal execution time for a 1D revolve algorithm.

    Notes
    -----
    In this case, the optimal solution is given as a function of the `cm`, `l`,
    `wd` and `rd`. Additional details about the execution time is avaiable in
    the paper [1], at the Theorem 3.15.

    [1] Aupy, G.,  Herrmann, Ju. and Hovland, P. and Robert, Y. "Optimal
    multistage algorithm for adjoint computation". SIAM Journal on Scientific
    Computing, 38(3), C232-C255, (2016).
    DOI: 10.1145/347837.347846

    Returns
    -------
    list
        The optimal execution time for the Disk-Revolve algorithm.
    """
    if opt_0 is None:
        opt_0 = get_opt_0_table(lmax, cm, uf, ub)
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(lmax, cm, ub, uf, rd, one_read_disk,
                                  opt_0=opt_0)
    opt_inf = Table()
    if __name__ == '__main__' and print_table:
        opt_inf.set_to_print(print_table)
    opt_inf.append(ub)
    # Opt_inf[1] for cm
    if cm == 0:
        opt_inf.append(wd + uf + 2 * ub + rd)
    else:
        opt_inf.append(uf + 2 * ub)
    # Opt_inf[2...lmax] for cm
    for l in range(2, lmax + 1):  # noqa: E741
        if one_read_disk:
            min_aux = min([wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1]
                           for j in range(1, l)])
            opt_inf.append(min(opt_0[cm][l], min_aux))
        else:
            min_aux = min([wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1]
                           for j in range(1, l)])
            opt_inf.append(min(opt_0[cm][l], min_aux))
    return opt_inf


def disk_revolve(l, cm, rd, wd, fwd_cost, bwd_cost,      # noqa: E741
                 opt_0=None, opt_1d=None, opt_inf=None):
    """Disk-Revolve algorithm.

    Parameters
    ----------
    l : int
        The number of forward steps to execute in the AC graph.
    cm : int
        The number of checkpoints stored in memory.
    opt_0 : list, optional
        Optimal execution time for the revolve algorithm.
    opt_1d : list, optional
        Optimal execution time for the 1D revolve algorithm.
    opt_inf : list, optional
        Optimal execution time.

    Returns
    -------
        Disk-Revolve schedule.
    """
    params = revolver_parameters(wd, rd, fwd_cost, bwd_cost)
    parameters = dict(params)
    uf = parameters["uf"]
    ub = parameters["ub"]
    rd = parameters["rd"]
    wd = parameters["wd"]
    one_read_disk = parameters["one_read_disk"]

    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, uf, ub)
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(l, cm, ub, uf, rd, one_read_disk,
                                  opt_0=opt_0)
    if opt_inf is None:
        opt_inf = get_opt_inf_table(l, cm, uf, ub, rd, wd, one_read_disk,
                                    opt_0=opt_0, opt_1d=opt_1d)
    sequence = Sequence(Function("Disk-Revolve", l, cm),
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:  # noqa: E741
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory",  1))
        return sequence
    if l == 1:  # noqa: E741
        if cm == 0:
            sequence.insert(operation("Write_disk", 0))
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
            sequence.insert(operation("Discard_disk", 0))
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
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1]
                    for j in range(1, l)]
    else:
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1]
                    for j in range(1, l)]
    if min(list_mem) < opt_0[cm][l]:
        jmin = argmin(list_mem)
        sequence.insert(operation("Write_disk", 0))
        sequence.insert(operation("Forward", [0, jmin]))
        sequence.insert_sequence(
            disk_revolve(l - jmin, cm, rd, wd, uf, ub,
                         opt_0=opt_0, opt_1d=opt_1d,
                         opt_inf=opt_inf).shift(jmin)
        )
        sequence.insert(operation("Read_disk", 0))
        if one_read_disk:
            sequence.insert_sequence(
                revolve(jmin - 1, cm, rd, wd, uf, ub, opt_0=opt_0)
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
