# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr),
# Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains the implementation of the H-Revolve schedule.
"""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, argmin)
from .utils import revolver_parameters


def get_hopt_table(lmax, cvect, wvect, rvect, ub, uf):
    """ Compute the optimal hierarchical execution time
    for the H-Revolve algorithm.

    Parameters
    ----------
    lmax : int
        The maximal number of foward/adjoint steps.
    cvect : tuple
        A tuple with the number of slots to store in K levels.
    wvect : tuple
        A tuple with the cost of writing the checkpoint data in each of the K
        levels.
    rvect : tuple
        A tuple with the cost of reading the checkpoint data in each of the K
        levels.
    ub : float, optional
        The cost of advancing the adjoint over one step.
    uf : float
        The cost of advancing the forward over one step.

    Notes
    -----
    This computation uses a dynamic program.
    K is the number of levels in the hierarchy.
    The *checkpoint_schedules* uses two storage levels: RAM and disk.
    Thus, K = 2.
    For more details on execution time, refer to the work presented in [1],
    at section 3.1.

    [1] Herrmann, J. and Pallez (Aupy), G.. "H-Revolve: a framework
    for adjoint computation on synchronous hierarchical platforms."
    ACM Transactions on Mathematical Software (TOMS) 46.2 (2020): 1-25.
    DOI: 10.1145/3378672.


    Returns
    -------
    tuple : (list, list)
        A tuple containing the execution time on such arquitecture.
    """
    K = len(cvect)
    assert len(wvect) == len(rvect) == len(cvect)
    opt = [[[float("inf")] * (cvect[i] + 1)
            for _ in range(lmax + 1)] for i in range(K)]
    optp = [[[float("inf")] * (cvect[i] + 1)
             for _ in range(lmax + 1)] for i in range(K)]
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
    for l in range(2, lmax + 1):  # noqa: E741
        optp[0][l][1] = (l + 1) * ub + l * (l + 1) / 2 * uf + l * rvect[0]
        opt[0][l][1] = wvect[0] + optp[0][l][1]
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):  # noqa: E741
            optp[0][l][m] = min([j * uf + opt[0][l - j][m - 1] + rvect[0] +
                                 optp[0][j - 1][m] for j in range(1, l)] + [optp[0][l][1]])  # noqa: E501
            opt[0][l][m] = wvect[0] + optp[0][l][m]
    # Fill K > 0
    for k in range(1, K):
        mmax = cvect[k]
        for l in range(2, lmax+1):  # noqa: E741
            opt[k][l][0] = opt[k-1][l][cvect[k-1]]
        for m in range(1, mmax + 1):
            for l in range(1, lmax + 1):  # noqa: E741
                optp[k][l][m] = min([opt[k-1][l][cvect[k-1]]] +
                                    [j * uf + opt[k][l - j][m - 1] + rvect[k]
                                     + optp[k][j - 1][m] for j in range(1, l)])
                opt[k][l][m] = min(opt[k-1][l][cvect[k-1]], wvect[k] + optp[k][l][m])  # noqa: E501
    return (optp, opt)


def hrevolve_aux(l, K, cmem, cvect, wvect, rvect, hoptp=None,  # noqa: E741
                 hopt=None, **params):
    """Auxiliary function to compute the H-Revolve sequence of operations.

    Parameters
    ----------
    l : int
        The number of forward steps to use in the AC(Adjoint Computation)
        graph.
    K : int
        Memory level, where `K = 0` represents RAM and `K = 1` represents disk.
    cmem : int
        Number of available slots in the K-th level of memory.
    cvect : tuple
        A tuple containing the maximal number of slots that must be stored in
        each level.
    wvect : tuple
        A tuple containing the cost of writing the checkpoint data in each
        level.
    rvect : tuple
        A tuple containing the cost of reading the checkpoint data in each
        level.
    hoptp : list
        Execution time for a optimal solution in which the data at step 0 is
        stored in the top K-th level of storage.
    hopt : list
        Execution time for general hierarchical AC problem.
    params : dict
        Input parameters to be passed to the `Op` function.

    Returns
    -------
    Sequence
        A sequence of operations.
    """
    uf = params["uf"]
    ub = params["ub"]
    if (hoptp is None) or (hopt is None):
        (hoptp, hopt) = get_hopt_table(l, cvect, wvect, rvect, uf, ub)
    sequence = Sequence(Function("hrevolve_aux", l, [K, cmem]),
                        levels=len(cvect), concat=params["concat"])
    operation = partial(Op, params=params)
    if cmem == 0:
        raise KeyError("hrevolve_aux should not be call with cmem = 0. Contact\
                       developers.")
    if l == 0:  # noqa: E741
        sequence.insert(operation("Write_Forward", [0, 1]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward", [0, 1]))
        return sequence
    if l == 1:  # noqa: E741
        if wvect[0] + rvect[0] < rvect[K]:
            sequence.insert(operation("Write", [0, 0]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Write_Forward", [0, 2]))
        sequence.insert(operation("Forward", [1, 2]))
        sequence.insert(operation("Backward", [2, 1]))
        sequence.insert(operation("Discard_Forward", [0, 2]))
        if wvect[0] + rvect[0] < rvect[K]:
            sequence.insert(operation("Read", [0, 0]))
        else:
            sequence.insert(operation("Read", [K, 0]))
        sequence.insert(operation("Write_Forward", [0, 1]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward", [0, 1]))
        sequence.insert(operation("Discard", [0, 0]))
        return sequence
    if K == 0 and cmem == 1:
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(operation("Read", [0, 0]))
            if index + 1 != 0:
                sequence.insert(operation("Forward", [0, index + 1]))
            sequence.insert(operation("Write_Forward", [0, index + 2]))
            sequence.insert(operation("Forward", [index + 1, index + 2]))
            sequence.insert(operation("Backward", [index + 2, index + 1]))
            sequence.insert(operation("Discard_Forward", [0, index + 2]))
        sequence.insert(operation("Read", [0, 0]))
        sequence.insert(operation("Write_Forward", [0, 1]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward", [0, 1]))
        sequence.insert(operation("Discard", [0, 0]))
        return sequence
    if K == 0:
        list_mem = [j * uf + hopt[0][l - j][cmem - 1] + rvect[0] +
                    hoptp[0][j - 1][cmem] for j in range(1, l)]
        if min(list_mem) < hoptp[0][l][1]:
            jmin = argmin(list_mem)
            sequence.insert(operation("Forward", [0, jmin]))
            sequence.insert_sequence(
                hrevolve_recurse(l - jmin, 0, cmem - 1, cvect, wvect, rvect,
                                 hoptp=hoptp, hopt=hopt, **params).shift(jmin)
            )
            sequence.insert(operation("Read", [0, 0]))
            sequence.insert_sequence(
                hrevolve_aux(jmin - 1, 0, cmem, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
            )
            aux = sequence
            while aux.type == 'Function':
                aux = aux.sequence[-1]
            if aux.type != "Discard":
                sequence.insert(operation("Discard", [0, 0]))
            return sequence
        else:
            sequence.insert_sequence(
                hrevolve_aux(l, 0, 1, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
            )
            return sequence
    list_mem = [j * uf + hopt[K][l - j][cmem - 1] + rvect[K] +
                hoptp[K][j - 1][cmem] for j in range(1, l)]
    if min(list_mem) < hopt[K-1][l][cvect[K-1]]:
        jmin = argmin(list_mem)
        sequence.insert(operation("Forward", [0, jmin]))
        sequence.insert_sequence(
            hrevolve_recurse(l - jmin, K, cmem - 1, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params).shift(jmin)
        )

        sequence.insert(operation("Read", [K, 0]))
        sequence.insert_sequence(
            hrevolve_aux(jmin - 1, K, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **params)
        )
        return sequence
    else:
        sequence.insert_sequence(
            hrevolve_recurse(l, K-1, cvect[K-1], cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
        )
        return sequence


def hrevolve(l, cvect, wvect, rvect, fwd_cost, bwd_cost):  # noqa: E741
    """H-Revolve algorithm.

    Parameters
    ----------
    l : int
        The number of forward steps in the initial forward calculation.
    cvect : tuple
        A tuple containing the number of slots in each storage level.
    wvect : tuple
        A tuple containing the cost of writing the checkpoint data in each
        storage level.
    rvect : tuple
        A tuple containing the cost of reading the checkpoint data in each
        storage level.

    Notes
    -----
    K is the number of levels in the hierarchy, where K = 2 for the two
    storage levels: `'RAM'` and `'disk'`.
    For more details on H-Revolve and its schedules, refer to the work
    presented in [1].

    [1] Herrmann, J. and Pallez (Aupy), G.. "H-Revolve: a framework for adjoint
    computation on synchronous hierarchical platforms." ACM Transactions on
    Mathematical Software (TOMS) 46.2 (2020): 1-25.
    DOI: 10.1145/3378672.

    Returns
    -------
    Sequence
        H-Revolve schedules.
    """
    params = revolver_parameters(wvect, rvect, fwd_cost, bwd_cost)
    h_rev = hrevolve_recurse(l, len(cvect)-1, cvect[-1], cvect, wvect, rvect,
                             hoptp=None, hopt=None, **params)

    return h_rev


def hrevolve_recurse(l, K, cmem, cvect, wvect, rvect, hoptp=None,  # noqa: E741
                     hopt=None, **params):
    """Hrevolve recurse schedule.

    Parameters
    ----------
    l : int
        Total number of forward step.
    K : int
        The level of memory. In a two-level memory (`'RAM'` and `'disk'`)
        setup, `K = 1` represents disk, and `K = 0` represents `'RAM'`.
    cmem : int
        Number of available slots in the K-th level of memory. For two-level
        memory, `cmem` represents the number of checkpoints that should be
        stored in disk.
    cvect : tuple
        The number of slots in each level of memory.
    wvect : tuple
        The cost of writing to each level of memory.
    rvect : tuple
        The cost of reading from each level of memory.

    Returns
    -------
    Sequence
        A sequence of operations.
    """
    parameters = dict(params)
    uf = params["uf"]
    ub = params["ub"]
    if (hoptp is None) or (hopt is None):
        (hoptp, hopt) = get_hopt_table(l, cvect, wvect, rvect, uf, ub)
    sequence = Sequence(Function("HRevolve", l, [K, cmem]),
                        levels=len(cvect), concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if l == 0:  # noqa: E741
        sequence.insert(operation("Write_Forward", [0, 1]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward", [0, 1]))
        return sequence
    if K == 0 and cmem == 0:
        raise KeyError("It's impossible to execute an AC graph of size > 0\
                       with no memory.")
    if l == 1:  # noqa: E741
        sequence.insert(operation("Write", [0, 0]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Write_Forward", [0, 2]))
        sequence.insert(operation("Forward", [1, 2]))
        sequence.insert(operation("Backward", [2, 1]))
        sequence.insert(operation("Discard_Forward", [0, 2]))
        sequence.insert(operation("Read", [0, 0]))
        sequence.insert(operation("Write_Forward", [0, 1]))
        sequence.insert(operation("Forward", [0, 1]))
        sequence.insert(operation("Backward", [1, 0]))
        sequence.insert(operation("Discard_Forward", [0, 1]))
        sequence.insert(operation("Discard", [0, 0]))
        return sequence
    if K == 0:
        sequence.insert(operation("Write", [0, 0]))
        sequence.insert_sequence(
            hrevolve_aux(l, 0, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence

    if wvect[K] + hoptp[K][l][cmem] < hopt[K-1][l][cvect[K-1]]:
        sequence.insert(operation("Write", [K, 0]))
        sequence.insert_sequence(
            hrevolve_aux(l, K, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence
    else:
        sequence.insert_sequence(
            hrevolve_recurse(l, K-1, cvect[K-1], cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence
