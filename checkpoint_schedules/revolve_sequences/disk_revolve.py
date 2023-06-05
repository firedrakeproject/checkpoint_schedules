#!/usr/bin/python
"""This function is a copy of the original disk_revolve
module that composes the python H-Revolve implementation
published by Herrmann and Pallez [1].

Refs:
[1] Herrmann, Pallez, "H-Revolve: A Framework for
Adjoint Computation on Synchronous Hierarchical
Platforms", ACM Transactions on Mathematical
Software  46(2), 2020.
"""
from .basic_functions import (Operation as Op, Table, Sequence, Function, argmin)
from .revolve import get_opt_0_table, revolve
from .revolve_1d import revolve_1d, get_opt_1d_table
from functools import partial


def get_opt_inf_table(lmax, cm, uf, ub, rd, wd, one_read_disk, print_table=None,
                      opt_0=None, opt_1d=None, **params):
    """Compute the opt_inf table for architecture and l=0...lmax.

    Parameters
    ----------
    lmax : int
        Maximal step.
    cm : int
        The number of checkpoint stored in memory.
    ub : float
        Cost of the backward steps.
    uf : float
        Cost of the forward steps.
    rd : _type_
        _description_
    wd : _type_
        _description_
    one_read_disk : _type_
        _description_
    print_table : _type_
        _description_
    opt_0 : _type_, optional
        _description_, by default None
    opt_1 : _type_, optional
        _description_, by default None

    Notes
    -----
    This computation uses a dynamic program.

    Returns
    -------
    
    """
    if opt_0 is None:
        opt_0 = get_opt_0_table(lmax, cm, **params)
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(lmax, cm, opt_0=opt_0, **params)
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
    for l in range(2, lmax + 1):
        if one_read_disk:
            opt_inf.append(min(opt_0[cm][l], min([wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1] for j in range(1, l)])))
        else:
            opt_inf.append(min(opt_0[cm][l], min([wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1] for j in range(1, l)])))
    return opt_inf

    
def disk_revolve(l, cm, opt_0=None, opt_1d=None,
                 opt_inf=None, **params):
    """Return a disk revolve sequence.

    Parameters
    ----------
    l : int
        Number of forward step to execute in the AC graph.
    cm : int
        Number of available memory slots.
    opt_0 : _type_, optional
        _description_, by default None
    opt_1d : _type_, optional
        _description_, by default None
    opt_inf : _type_, optional
        _description_, by default None

    Returns
    -------

        Return the optimal sequence of makespan Opt_inf(l, cm).
    """

    parameters = dict(params)
    uf = parameters["uf"]
    rd = parameters["rd"]
    wd = parameters["wd"]
    one_read_disk = parameters["one_read_disk"]

    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, **parameters)
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(l, cm, opt_0=opt_0, **parameters)
    if opt_inf is None:
        opt_inf = get_opt_inf_table(l, cm, opt_0=opt_0, opt_1d=opt_1d,
                                    **parameters)
    sequence = Sequence(Function("Disk-Revolve", l, cm), concat=parameters["concat"])
    Operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(Operation("Write_Forward_memory", 1))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Backward", [1, 0]))
        sequence.insert(Operation("Discard_Forward_memory",  1))
        return sequence
    if l == 1:
        if cm == 0:
            sequence.insert(Operation("Write_disk", 0))
            sequence.insert(Operation("Forward", [0, 1]))
            sequence.insert(Operation("Write_Forward_memory", 2))
            sequence.insert(Operation("Forward", [1, 2]))
            sequence.insert(Operation("Backward", [2, 1]))
            sequence.insert(Operation("Discard_Forward_memory", 2))
            sequence.insert(Operation("Read_disk", 0))
            sequence.insert(Operation("Write_Forward_memory", 1))
            sequence.insert(Operation("Forward", [0, 1]))
            sequence.insert(Operation("Backward", [1, 0]))
            sequence.insert(Operation("Discard_Forward_memory", 1))
            sequence.insert(Operation("Discard_disk", 0))
            return sequence
        else:
            sequence.insert(Operation("Write_memory", 0))
            sequence.insert(Operation("Forward", [0, 1]))
            sequence.insert(Operation("Write_Forward_memory", 2))
            sequence.insert(Operation("Forward", [1, 2]))
            sequence.insert(Operation("Backward", [2, 1]))
            sequence.insert(Operation("Discard_Forward_memory", 2))
            sequence.insert(Operation("Read_memory", 0))
            sequence.insert(Operation("Write_Forward_memory", 1))
            sequence.insert(Operation("Forward", [0, 1]))
            sequence.insert(Operation("Backward", [1, 0]))
            sequence.insert(Operation("Discard_Forward_memory", 1))
            sequence.insert(Operation("Discard_memory", 0))
            return sequence
    if one_read_disk:
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1] for j in range(1, l)]
    else:
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1] for j in range(1, l)]
    if min(list_mem) < opt_0[cm][l]:
        jmin = argmin(list_mem)
        sequence.insert(Operation("Write_disk", 0))
        sequence.insert(Operation("Forward", [0, jmin]))
        sequence.insert_sequence(
            disk_revolve(l - jmin, cm, opt_0=opt_0, opt_1d=opt_1d, opt_inf=opt_inf, **parameters).shift(jmin)
        )
        sequence.insert(Operation("Read_disk", 0))
        if one_read_disk:
            sequence.insert_sequence(
                revolve(jmin - 1, cm, opt_0=opt_0, **parameters)
            )
        else:
            sequence.insert_sequence(
                revolve_1d(jmin - 1, cm, opt_0=opt_0, opt_1d=opt_1d, **parameters)
            )
        return sequence
    else:
        sequence.insert_sequence(revolve(l, cm, opt_0=opt_0, **parameters))
        return sequence
