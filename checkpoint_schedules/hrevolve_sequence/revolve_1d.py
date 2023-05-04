#!/usr/bin/python

from .parameters import defaults
from .basic_functions import (Operation as Op, Sequence, Function, Table,
                              my_buddy, argmin)
from .revolve import revolve, get_opt_0_table
from functools import partial


def get_opt_1d_table(lmax, cm, print_table, ub, uf, rd, one_read_disk,
                     opt_0=None, **params):
    """ Compute the opt_1d table for cm and l=0...lmax
        This computation uses a dynamic program
        We consider that x_0 is already stored on the disk"""
    if opt_0 is None:
        opt_0 = get_opt_0_table(lmax, cm)
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
            opt_1d.append(min(opt_0[cm][l], min([j * uf + opt_0[cm][l - j] + rd + opt_0[cm][j-1] for j in range(1, l)])))
        else:
            opt_1d.append(min(opt_0[cm][l], min([j * uf + opt_0[cm][l - j] + rd + opt_1d[j-1] for j in range(1, l)])))
    return opt_1d


def revolve_1d(l, cm, opt_0=None, opt_1d=None, **params):
    """ l : number of forward step to execute in the AC graph
        cm : number of available memory slots
        Return the optimal sequence of makespan Opt_1d(l, cm)
        We consider that x_0 is already stored on the disk"""
    parameters = dict(defaults)
    parameters.update(params)
    rd = parameters["rd"]
    uf = parameters["uf"]
    one_read_disk = parameters["one_read_disk"]
    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, **parameters)
    if opt_1d is None:
        opt_1d = get_opt_1d_table(l, cm, opt_0=opt_0, **parameters)
    sequence = Sequence(Function("1D-Revolve", l, cm), concat=parameters["concat"])
    Operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(Operation("Backward", my_buddy(-1, l-1)))
        return sequence
    if l == 1:
        if cm == 0:
            sequence.insert(Operation("Forward", 0))
            sequence.insert(Operation("Backward", 1))
            sequence.insert(Operation("Read_disk", 0))
            sequence.insert(Operation("Backward", 0))
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
        list_mem = [j * uf + opt_0[cm][l - j] + rd + opt_0[cm][j-1] for j in range(1, l)]
    else:
        list_mem = [j * uf + opt_0[cm][l - j] + rd + opt_1d[j-1] for j in range(1, l)]
    if min(list_mem) < opt_0[cm][l]:
        jmin = argmin(list_mem)
        sequence.insert(Operation("Forward", [0, jmin - 1]))
        sequence.insert_sequence(
            revolve(l - jmin, cm, opt_0=opt_0, **parameters).shift(jmin)
        )
        sequence.insert(Operation("Read_disk", 0))
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
