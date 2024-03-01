# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr),
# Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains the functions used to compute the revolver sequences.
"""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, Table,
                              argmin)
from .utils import revolver_parameters


def get_opt_0_table(lmax, mmax, uf, ub, print_table=None):
    """Compute optimal execution time for Revolve algorithm.

    Parameters
    ----------
    lmax : int
        The number of forward steps to use in the AC graph.
    mmax : int
        Slots number in memory.
    ub : float, optional
        The cost of advancing the adjoint over one step.
    uf : float
        The cost of advancing the forward over one step.
    print_table : str, optional
        File to which to print the results table.

    Notes
    -----
    The computation uses a dynamic program.

    Returns
    -------
    list
        Optimal execution time for Revolve algorithm.
    """
    # Build table
    opt = [Table() for _ in range(mmax + 1)]
    if __name__ == '__main__' and print_table:
        opt[mmax].set_to_print(print_table)
    # Initialize borders of the tables
    for m in range(mmax + 1):
        opt[m].append(ub)
    for m in range(1, mmax + 1):
        opt[m].append(uf + 2 * ub)
    for l in range(2, lmax + 1):  # noqa: E741
        opt[1].append((l+1) * ub + l * (l + 1) / 2 * uf)
    # Compute everything
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):  # noqa: E741
            value = min([j * uf + opt[m-1][l - j] + opt[m][j-1]
                         for j in range(1, l)])
            opt[m].append(value)
    return opt


def revolve(l, cm, rd, wd, fwd_cost, bwd_cost, opt_0=None):  # noqa: E741
    """Return a revolve sequence.

    Parameters
    ----------
    l : int
        Number of forward step to execute in the AC graph.
    cm : int
        The number of checkpoints stored in memory.
    opt_0 : list, optional
        Return the optimal sequence of makespan.

    Returns
    -------
    Sequence
        Revolve schedule
    """
    params = revolver_parameters(wd, rd, fwd_cost, bwd_cost)
    parameters = dict(params)
    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, fwd_cost, bwd_cost)
    sequence = Sequence(Function("Revolve", l, cm),
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:  # noqa: E741
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory", 1))
        sequence.insert(operation("Discard_memory", 0))
        return sequence
    elif cm == 0:
        raise ValueError("It's impossible to execute an AC graph without\
                         memory")
    elif l == 1:  # noqa: E741
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
    elif cm == 1:
        sequence.insert(operation("Write_memory", 0))
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(operation("Read_memory", 0))
            if index + 1 != 0:
                sequence.insert(operation("Forward", [0, index + 1]))
            sequence.insert(operation("Write_Forward_memory", index + 2))
            sequence.insert(operation("Forward", [index + 1, index + 2]))
            sequence.insert(operation("Backward", [index + 2, index + 1]))
            sequence.insert(operation("Discard_Forward_memory", index + 2))
        sequence.insert(operation("Read_memory", 0))
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory", 1))
        sequence.insert(operation("Discard_memory", 0))
        return sequence
    list_mem = [j*parameters["uf"] + opt_0[cm-1][l-j] + opt_0[cm][j-1]
                for j in range(1, l)]
    jmin = argmin(list_mem)
    sequence.insert(operation("Write_memory", 0))
    sequence.insert(operation("Forward", [0, jmin]))
    sequence.insert_sequence(
        revolve(l - jmin, cm - 1, wd, rd, fwd_cost, bwd_cost,
                opt_0=opt_0).shift(jmin)
    )
    sequence.insert(operation("Read_memory", 0))
    sequence.insert_sequence(
        revolve(jmin - 1, cm, wd, rd, fwd_cost,
                bwd_cost, opt_0=opt_0).remove_useless_wm()
    )
    return sequence
