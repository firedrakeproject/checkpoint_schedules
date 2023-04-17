#!/usr/bin/python
"""
This module includes classes and function definitions
provided as part of the H-Revolve python implementation.
The original implementation was developed by authors
Julien Herrmann and Guillaume Aupy and is orignally
distributed under GNU GPL v.3 license terms.
The original H-Revolve source code can be found in the
following Gitlab repository:

Original H-Revolve source-code:
https://gitlab.inria.fr/adjoint-computation/H-Revolve/tree/master

The H-Revolve library is described in detail in the
paper "H-Revolve: A Framework for Adjoint Computation on
Synchronous Hierarchical Platforms" by Herrmann and Pallez [1].

Some minor modifications where made to adapt this libray for the
checkpoint schedules API.

Authors: Julien Herrmann, Guillaume Aupy

Refs:
[1] Herrmann, Pallez, "H-Revolve: A Framework for
    Adjoint Computation on Synchronous Hierarchical
    Platforms", ACM Transactions on Mathematical
    Software  46(2), 2020.
"""
from .parameters import defaults
from .basic_functions import (Operation as Op, Sequence, Function, my_buddy,
                              argmin)
from functools import partial


def get_hopt_table(lmax, cvect, wvect, rvect, ub=1, uf=1, **params):
    """Compute the HOpt table for architecture and l=0...lmax.
        This computation uses a dynamic program.

    Parameters
    ----------
    lmax : int
        Total checkpoint of a forward solver.
    cvect : _type_
        _description_
    wvect : _type_
        _description_
    rvect : _type_
        _description_
    ub : _type_
        Cost of the forward steps, by default 1.
    uf : _type_
        Cost of the backward steps, by default 1.

    Returns
    -------
    tuple(list, list)
        _description_

    """
    K = len(cvect)
    assert len(wvect) == len(rvect) == len(cvect)
    opt = [[[float("inf")] * (cvect[i] + 1) for _ in range(lmax + 1)] for i in range(K)]
    optp = [[[float("inf")] * (cvect[i] + 1) for _ in range(lmax + 1)] for i in range(K)]
    # Initialize borders of the table
    for k in range(K):
        mmax = cvect[k]
        for m in range(mmax + 1):
            opt[k][0][m] = ub
            optp[k][0][m] = ub
        for m in range(mmax + 1):
            if (m == 0) and (k == 0):
                continue
            optp[k][1][m] = uf + 2 * ub + rvect[0]
            opt[k][1][m] = wvect[0] + optp[k][1][m]
    # Fill K = 0
    mmax = cvect[0]
    for l in range(2, lmax + 1):
        optp[0][l][1] = (l + 1) * ub + l * (l + 1) / 2 * uf + l * rvect[0]
        opt[0][l][1] = wvect[0] + optp[0][l][1]
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):
            optp[0][l][m] = min([j * uf + opt[0][l - j][m - 1] + rvect[0] + optp[0][j - 1][m] for j in range(1, l)] + [optp[0][l][1]])
            opt[0][l][m] = wvect[0] + optp[0][l][m]
    # Fill K > 0
    for k in range(1, K):
        mmax = cvect[k]
        for l in range(2, lmax+1):
            opt[k][l][0] = opt[k-1][l][cvect[k-1]]
        for m in range(1, mmax + 1):
            for l in range(1, lmax + 1):
                optp[k][l][m] = min([opt[k-1][l][cvect[k-1]]] + [j * uf + opt[k][l - j][m - 1] + rvect[k] + optp[k][j - 1][m] for j in range(1, l)])
                opt[k][l][m] = min(opt[k-1][l][cvect[k-1]], wvect[k] + optp[k][l][m])
    return (optp, opt)


def HRevolve_aux(l, K, cmem, cvect, wvect, rvect, hoptp=None, hopt=None, **params):
    r""" l : number of forward step to execute in the AC graph
            K: the level of memory
            cmem: number of available slots in the K-th level of memory
            Return the optimal sequence of makespan \overline{HOpt}(l, architecture)"""
    uf = params["uf"]
    if (hoptp is None) or (hopt is None):
        (hoptp, hopt) = get_hopt_table(l, cvect, wvect, rvect, **params)
    sequence = Sequence(Function("HRevolve_aux", l, [K, cmem]),
                        levels=len(cvect), concat=params["concat"])
    Operation = partial(Op, params=params)
    if cmem == 0:
        raise KeyError("HRevolve_aux should not be call with cmem = 0. Contact developers.")
    if l == 0:
        sequence.insert(Operation("Backward", my_buddy(-1)))
        return sequence
    if l == 1:
        if wvect[0] + rvect[0] < rvect[K]:
            sequence.insert(Operation("Write", [0, 0]))
        sequence.insert(Operation("Forward", 0))
        sequence.insert(Operation("Backward", my_buddy(0)))
        if wvect[0] + rvect[0] < rvect[K]:
            sequence.insert(Operation("Read", [0, 0]))
        else:
            sequence.insert(Operation("Read", [K, 0]))
        sequence.insert(Operation("Backward", my_buddy(-1)))
        sequence.insert(Operation("Discard", [0, 0]))
        return sequence
    if K == 0 and cmem == 1:
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(Operation("Read", [0, 0]))
            sequence.insert(Operation("Forwards", [0, index]))
            sequence.insert(Operation("Backward", my_buddy(index)))
        sequence.insert(Operation("Read", [0, 0]))
        sequence.insert(Operation("Backward", my_buddy(-1)))
        sequence.insert(Operation("Discard", [0, 0]))
        return sequence
    if K == 0:
        list_mem = [j * uf + hopt[0][l - j][cmem - 1] + rvect[0] + hoptp[0][j - 1][cmem] for j in range(1, l)]
        if min(list_mem) < hoptp[0][l][1]:
            jmin = argmin(list_mem)
            sequence.insert(Operation("Forwards", [0, jmin - 1]))
            sequence.insert_sequence(
                hrevolve_recurse(l - jmin, 0, cmem - 1, cvect, wvect, rvect,
                                 hoptp=hoptp, hopt=hopt, **params).shift(jmin)
            )
            sequence.insert(Operation("Read", [0, 0]))
            sequence.insert_sequence(
                HRevolve_aux(jmin - 1, 0, cmem, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
            )
            return sequence
        else:
            sequence.insert_sequence(
                HRevolve_aux(l, 0, 1, cvect, wvect, rvect, uf,
                             hoptp=hoptp, hopt=hopt, **params)
            )
            return sequence
    list_mem = [j * uf + hopt[K][l - j][cmem - 1] + rvect[K] + hoptp[K][j - 1][cmem] for j in range(1, l)]
    if min(list_mem) < hopt[K-1][l][cvect[K-1]]:
        jmin = argmin(list_mem)
        sequence.insert(Operation("Forwards", [0, jmin - 1]))
        sequence.insert_sequence(
            hrevolve_recurse(l - jmin, K, cmem - 1, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params).shift(jmin)
        )
        sequence.insert(Operation("Read", [K, 0]))
        sequence.insert_sequence(
            HRevolve_aux(jmin - 1, K, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **params)
        )
        return sequence
    else:
        sequence.insert_sequence(
            hrevolve_recurse(l, K-1, cvect[K-1], cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
        )
        return sequence


def hrevolve(l, cvect, wvect, rvect, **params):
    """ l : number of forward step to execute in the AC graph
        cvect: the number of slots in each level of memory
        wvect: the cost of writing to each level of memory
        rvect: the cost of reading from each level of memory
        Return the optimal sequence of makespan HOpt(l, architecture)"""
    params["wd"] = wvect
    params["rd"] = rvect
    return hrevolve_recurse(l, len(cvect)-1, cvect[-1], cvect, wvect, rvect,
                            hoptp=None, hopt=None, **params)


def hrevolve_recurse(l, K, cmem, cvect, wvect, rvect, hoptp=None, hopt=None, **params):
    """ l : number of forward step to execute in the AC graph
        K: the level of memory
        cmem: number of available slots in the K-th level of memory
        cvect: the number of slots in each level of memory
        wvect: the cost of writing to each level of memory
        rvect: the cost of reading from each level of memory
        Return the optimal sequence of makespan HOpt(l, architecture)"""
   
    parameters = dict(defaults)
    parameters.update(params)
    if (hoptp is None) or (hopt is None):
        (hoptp, hopt) = get_hopt_table(l, cvect, wvect, rvect, **parameters)
    sequence = Sequence(Function("HRevolve", l, [K, cmem]),
                        levels=len(cvect), concat=parameters["concat"])
    Operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(Operation("Backward", my_buddy(-1)))
        return sequence
    if K == 0 and cmem == 0:
        raise KeyError("It's impossible to execute an AC graph of size > 0 with no memory.")
    if l == 1:
        sequence.insert(Operation("Write", [0, 0]))
        sequence.insert(Operation("Forward", 0))
        sequence.insert(Operation("Backward", my_buddy(0)))
        sequence.insert(Operation("Read", [0, 0]))
        sequence.insert(Operation("Backward", my_buddy(-1)))
        sequence.insert(Operation("Discard", [0, 0]))
        return sequence
    if K == 0:
        sequence.insert(Operation("Write", [0, 0]))
        sequence.insert_sequence(
            HRevolve_aux(l, 0, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence
    if wvect[K] + hoptp[K][l][cmem] < hopt[K-1][l][cvect[K-1]]:
        sequence.insert(Operation("Write", [K, 0]))
        sequence.insert_sequence(
            HRevolve_aux(l, K, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence
    else:
        sequence.insert_sequence(
            hrevolve_recurse(l, K-1, cvect[K-1], cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence
