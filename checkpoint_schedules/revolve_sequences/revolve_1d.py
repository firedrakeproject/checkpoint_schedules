#!/usr/bin/python
"""This function is a copy of the original revolve_1d
module that composes the python H-Revolve implementation
published by Herrmann and Pallez [1].

Refs:
[1] Herrmann, Pallez, "H-Revolve: A Framework for
Adjoint Computation on Synchronous Hierarchical
Platforms", ACM Transactions on Mathematical
Software  46(2), 2020.
"""
from .basic_functions import (Operation as Op, Sequence, Function, Table, argmin)
from .revolve import revolve, get_opt_0_table
from functools import partial


def get_opt_1d_table(lmax, cm, ub, uf, rd, one_read_disk, print_table=None,
                     opt_0=None, **params):
    """Compute the opt_1d table. 
    This computation uses a dynamic program.

    Parameters
    ----------
    lmax : int
        Maximal step.
    cm : int
        The number of checkpoint stored in memory.
    print_table : _type_
        _description_
    ub : float
        Cost of the backward steps.
    uf : float
        Cost of the forward steps.
    rd : _type_
        _description_
    one_read_disk : _type_
        _description_
    opt_0 : _type_, optional
        _description_, by default None

    Notes
    -----
    Consider that 'x_0' is already stored on the disk.

    Returns
    -------
    _type_
        _description_
    """
    if opt_0 is None:
        opt_0 = get_opt_0_table(lmax, cm, uf, ub, print_table, **params)
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
    for l in range(2, lmax + 1):
        if one_read_disk:
            m = min([j * uf + opt_0[cm][l - j] + rd + opt_0[cm][j-1] for j in range(1, l)])
            opt_1d.append(min(opt_0[cm][l], m))
        else:
            m = min([j * uf + opt_0[cm][l - j] + rd + opt_1d[j-1] for j in range(1, l)])
            opt_1d.append(min(opt_0[cm][l], m))
    return opt_1d


def revolve_1d(l, cm, opt_0=None, opt_1d=None, **params):
    """Return the 1D revolve sequence.

    Parameters
    ----------
    l : int
        Total number of the forward step.
    cm : int
        Number of available memory slots.
    opt_0 : _type_, optional
        _description_
    opt_1d : _type_, optional
        _description_

    Notes
    -----
    Consider that "x_0" is already stored on the disk.

    Returns
    -------
    object
        1D revolve sequence.
    """
    parameters = dict(params)
    rd = parameters["rd"]
    uf = parameters["uf"]
    one_read_disk = parameters["one_read_disk"]
    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, **parameters)
    if opt_1d is None:
        opt_1d = get_opt_1d_table(l, cm, opt_0=opt_0, **parameters)
    sequence = Sequence(Function("1D-Revolve", l, cm), concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory", 1))
        return sequence
    if l == 1:
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
        list_mem = [j * uf + opt_0[cm][l - j] + rd + opt_0[cm][j-1] for j in range(1, l)]
    else:
        list_mem = [j * uf + opt_0[cm][l - j] + rd + opt_1d[j-1] for j in range(1, l)]
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
        sequence.insert_sequence(revolve(l, cm, opt_0=opt_0, **parameters))
        return sequence
