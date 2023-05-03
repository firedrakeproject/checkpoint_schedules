#!/usr/bin/python
from .parameters import defaults
from .basic_functions import (Operation as Op, Sequence, Function, Table, beta,
                              my_buddy, argmin)
from functools import partial


def get_t(l, cm):
    t = 0
    while (beta(cm, t) <= l):
        t += 1
    return t-1


def opt_0_closed_formula(l, cm, uf, ub, **params):
    """ Fast computation of Opt_0 based on the closed formula """
    if l == 0:
        return ub
    t = get_t(l, cm)
    return ((l+1) * (t+1) - beta(cm+1, t)) * uf + (l+1) * ub


def get_opt_0_table(lmax, mmax, uf, ub, print_table, **params):
    """ Return the Opt_0 tables
        for every Opt_0[m][l] with l = 0...lmax and m = 0...mmax
        The computation uses a dynamic program"""
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
            value = min([j * uf + opt[m-1][l - j] + opt[m][j-1] for j in range(1, l)])
            opt[m].append(value)
    return opt


def revolve(l, cm, opt_0=None, **params):
    """ l : number of forward step to execute in the AC graph
        cm : number of available memory slots
        Return the optimal sequence of makespan Opt_0(l, cm)"""
    parameters = dict(defaults)
    parameters.update(params)
    if opt_0 is None:
        opt_0 = get_opt_0_table(l, cm, **params)
    sequence = Sequence(Function("Revolve", l, cm), concat=parameters["concat"])
    Operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(Operation("Backward", my_buddy(-1, l-1)))
        sequence.insert(Operation("Discard_memory", 0))
        return sequence
    elif cm == 0:
        raise ValueError("It's impossible to execute an AC graph without memory")
    elif l == 1:
        sequence.insert(Operation("Write_memory", 0))
        sequence.insert(Operation("Forward", 0))
        sequence.insert(Operation("Backward", my_buddy(0, l-1)))
        sequence.insert(Operation("Read_memory", 0))
        sequence.insert(Operation("Backward", my_buddy(-1, l-1)))
        sequence.insert(Operation("Discard_memory", 0))
        return sequence
    elif cm == 1:
        sequence.insert(Operation("Write_memory", 0))
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(Operation("Read_memory", 0))
            sequence.insert(Operation("Forwards", [0, index]))
            sequence.insert(Operation("Backward", my_buddy(index, l-1)))
        sequence.insert(Operation("Read_memory", 0))
        sequence.insert(Operation("Backward", my_buddy(-1, l-1)))
        sequence.insert(Operation("Discard_memory", 0))
        return sequence
    list_mem = [j*parameters["uf"] + opt_0[cm-1][l-j] + opt_0[cm][j-1] for j in range(1, l)]
    jmin = argmin(list_mem)
    sequence.insert(Operation("Write_memory", 0))
    sequence.insert(Operation("Forwards", [0, jmin-1]))
    sequence.insert_sequence(
        revolve(l - jmin, cm - 1, opt_0=opt_0, **parameters).shift(jmin)
    )
    sequence.insert(Operation("Read_memory", 0))
    sequence.insert_sequence(
        revolve(jmin-1, cm, opt_0=opt_0, **parameters).remove_useless_wm()
    )
    return sequence
