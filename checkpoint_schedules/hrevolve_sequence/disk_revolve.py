#!/usr/bin/python

from .parameters import defaults
from .basic_functions import (Operation as Op, Table, Sequence, Function, argmin)
from .revolve import get_opt_0_table, revolve
from .revolve_1d import revolve_1d, get_opt_1d_table
from functools import partial


def get_opt_inf_table(lmax, cm, uf, ub, rd, wd, one_read_disk, print_table,
                      opt_0=None, opt_1d=None, **params):
    """ Compute the opt_inf table for cm and l=0...lmax
        This computation uses a dynamic program"""
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
    """ l : number of forward step to execute in the AC graph
        cm : number of available memory slots
        Return the optimal sequence of makespan Opt_inf(l, cm)"""
    
    parameters = dict(defaults)
    parameters.update(params)
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
        sequence.insert(Operation("Backward", 0))
        return sequence
    if l == 1:
        if cm == 0:
            sequence.insert(Operation("Write_disk", 0))
            sequence.insert(Operation("Forward", 0))
            sequence.insert(Operation("Backward", 1))
            sequence.insert(Operation("Read_disk", 0))
            sequence.insert(Operation("Backward", 0))
            sequence.insert(Operation("Discard_disk", 0))
            return sequence
        else:
            sequence.insert(Operation("Write_memory", 0))
            sequence.insert(Operation("Forward", 0))
            sequence.insert(Operation("Backward", 1))
            sequence.insert(Operation("Read_memory", 0))
            sequence.insert(Operation("Backward", 0))
            sequence.insert(Operation("Discard_memory", 0))
            return sequence
    if one_read_disk:
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_0[cm][j-1] for j in range(1, l)]
    else:
        list_mem = [wd + j * uf + opt_inf[l - j] + rd + opt_1d[j-1] for j in range(1, l)]
    if min(list_mem) < opt_0[cm][l]:
        jmin = argmin(list_mem)
        sequence.insert(Operation("Write_disk", 0))
        sequence.insert(Operation("Forwards", [0, jmin - 1]))
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
