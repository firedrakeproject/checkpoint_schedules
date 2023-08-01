#!/usr/bin/python
"""This module contains the functions used to compute the revolver sequences."""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, Table, beta, argmin)
from .utils import revolver_parameters


def get_t(l, cm):
    """Return the smallest t.

    Parameters
    ----------
    l : int
        Steps number.
    cm : int
        The number of checkpoints stored in memory.

    Returns
    -------
    int
        .
    """
    t = 0
    while (beta(cm, t) <= l):
        t += 1
    return t-1


def opt_0_closed_formula(l, cm, uf, ub):
    """Fast computation of "Opt_0" based on the closed formula.

    Parameters
    ----------
    l : int
        The number of forward steps to use in the AC graph.
    cm : int
        The number of checkpoints stored in memory.
    uf : float
        The cost of advancing the forward over one step.
    ub : float
        The cost of advancing the adjoint over one step.

    Returns
    -------
    _type_
        _description_
    """
    if l == 0:
        return ub
    t = get_t(l, cm)
    return ((l+1) * (t+1) - beta(cm+1, t)) * uf + (l+1) * ub


def get_opt_0_table(lmax, mmax, uf, ub, print_table=None):
    """Return the Opt_0 tables.

    Parameters
    ----------
    lmax : int
        The number of forward steps to use in the AC graph.
    mmax : _type_
        _description_
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
    _type_
        _description_
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
    for l in range(2, lmax + 1):
        opt[1].append((l+1) * ub + l * (l + 1) / 2 * uf)
    # Compute everything
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):
            value = min([j * uf + opt[m-1][l - j] + opt[m][j-1] 
                         for j in range(1, l)])
            opt[m].append(value)
    return opt


def revolve(l, cm, rd, wd, fwd_cost, bwd_cost, opt_0=None):
    """Return a revolve sequence.

    Parameters
    ----------
    l : int
        Number of forward step to execute in the AC graph.
    cm : int
        The number of checkpoints stored in memory.
    opt_0 : _type_, optional
        Return the optimal sequence of makespan.

    Returns
    -------
    _type_
        _description_
    """
    params = revolver_parameters(wd, rd, fwd_cost, bwd_cost)
    parameters = dict(params)
    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, fwd_cost, bwd_cost)
    sequence = Sequence(Function("Revolve", l, cm), 
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory", 1))
        sequence.insert(operation("Discard_memory", 0))
        return sequence
    elif cm == 0:
        raise ValueError("It's impossible to execute an AC graph without memory")
    elif l == 1:
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
