#!/usr/bin/python
"""This module contains the functions to compute the Disk-Revolve schedules.
"""
from functools import partial
from .basic_functions import (Operation as Op, Table, Sequence, Function, argmin)
from .revolve import get_opt_0_table, revolve
from .revolve_1d import revolve_1d, get_opt_1d_table
from .utils import revolver_parameters


def get_opt_inf_table(lmax, cm, uf, ub, rd, wd, one_read_disk, 
                      print_table=None, opt_0=None, opt_1d=None, **params):
    """Compute the opt_inf table for architecture and l=0...lmax.

    Parameters
    ----------
    lmax : int
        The number of forward steps to use in the AC graph.
    cm : int
        The number of checkpoints stored in memory.
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
    opt_0 : _type_, optional
        _description_, by default None
    opt_1 : _type_, optional
        _description_, by default None
    **params : dict
        Dictionary of the parameters.

    Returns
    -------
    Table
        Return the opt_inf table for architecture and l=0...lmax.
    
    """
    if opt_0 is None:
        opt_0 = get_opt_0_table(lmax, cm, uf, ub)
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
            min_aux = min([wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1] for j in range(1, l)])
            opt_inf.append(min(opt_0[cm][l], min_aux))
        else:
            min_aux = min([wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1] for j in range(1, l)])
            opt_inf.append(min(opt_0[cm][l], min_aux))
    return opt_inf

    
def disk_revolve(l, cm, rd, wd, fwd_cost, bwd_cost,
                 opt_0=None, opt_1d=None, opt_inf=None):
    """Disk-Revolve algorithm.

    Parameters
    ----------
    l : int
        The number of forward steps to execute in the AC graph.
    cm : int
        The number of checkpoints stored in memory.
    opt_0 : 

    opt_1d :

    opt_inf : 

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
        opt_1d = get_opt_1d_table(l, cm, opt_0=opt_0, **parameters)
    if opt_inf is None:
        opt_inf = get_opt_inf_table(l, cm, opt_0=opt_0, opt_1d=opt_1d,
                                    **parameters)
    sequence = Sequence(Function("Disk-Revolve", l, cm), 
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(operation("Write_Forward_memory", 1))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward_memory",  1))
        return sequence
    if l == 1:
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
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1] for j in range(1, l)]
    else:
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1] for j in range(1, l)]
    if min(list_mem) < opt_0[cm][l]:
        jmin = argmin(list_mem)
        sequence.insert(operation("Write_disk", 0))
        sequence.insert(operation("Forward", [0, jmin]))
        sequence.insert_sequence(
            disk_revolve(l - jmin, cm, rd, wd, uf, ub, opt_0=opt_0, opt_1d=opt_1d, opt_inf=opt_inf).shift(jmin)
        )
        sequence.insert(operation("Read_disk", 0))
        if one_read_disk:
            sequence.insert_sequence(
                revolve(jmin - 1, cm, rd, wd, uf, ub, opt_0=opt_0)
            )
        else:
            sequence.insert_sequence(
                revolve_1d(jmin - 1, cm, opt_0=opt_0, opt_1d=opt_1d, **parameters)
            )
        return sequence
    else:
        sequence.insert_sequence(revolve(l, cm, rd, wd, uf, ub, opt_0=opt_0))
        return sequence
