"""Rotine of the H-Revolve schedules.
"""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, argmin)
from .utils import revolver_parameters


def get_hopt_table(lmax, cvect, wvect, rvect, ub, uf):
    """Compute the hierarchical AC problem which gives the 
    minimal makespan according the input parameters.

    Parameters
    ----------
    lmax : int
        The number of forward steps to execute in the AC graph.
    cvect : tuple
        The maximal number of slots that needs to be stored in the levels.
    wvect : tuple
        Number of elements defining the write cost associated with storing
        the checkpoint data used to restart the forward solver.
    rvect : tuple
        Number of elements defining the read cost associated with storing
        the checkpoint data used to restart the forward solver.
    ub : float, optional
        The cost of advancing the adjoint over that step.
    uf : float
        The cost of advancing the forward one step.

    Notes
    -----
    The term makespan is used for the total execution time.
    So, minimize makespan means minimize the execution time.

    Returns
    -------
    tuple : (list, list)
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


def hrevolve_aux(l, K, cmem, cvect, wvect, rvect, hoptp=None, hopt=None, **params):
    """Auxiliary function used to built the H-Revolve sequence.

    Parameters
    ----------
    l : int
        The number of forward steps to execute in the AC graph.
    K : int
        Memory level.
    cmem : int
        Number of available slots in the K-th level of memory.
        For instance, in two level of memory (RAM and DISK), `cmem` collects
        the number of checkpoints stored in DISK.
    cvect : tuple
        The maximal number of slots that needs to be stored in the levels.
    wvect : tuple
        Number of elements defining the write cost associated with storing
        the checkpoint data used to restart the forward solver.
    rvect : tuple
        Number of elements defining the read cost associated with storing
        the checkpoint data used to restart the forward solver.
    hoptp
        _description_ 
    hopt : _type_, optional
        _description_
    
    Returns
    -------
    tuple
        Return the optimal sequence of makespan.

    Raises
    ------
    KeyError
        If `cmem = 0`, `hrevolve_aux` should not be call.
    """
    uf = params["uf"]
    ub = params["ub"]
    if (hoptp is None) or (hopt is None):
        (hoptp, hopt) = get_hopt_table(l, cvect, wvect, rvect, uf, ub)
    sequence = Sequence(Function("hrevolve_aux", l, [K, cmem]),
                        levels=len(cvect), concat=params["concat"])
    Operation = partial(Op, params=params)
    if cmem == 0:
        raise KeyError("hrevolve_aux should not be call with cmem = 0. Contact developers.")
    if l == 0:
        sequence.insert(Operation("Write_Forward", [0, 1]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Backward", [1, 0]))
        sequence.insert(Operation("Discard_Forward", [0, 1]))
        return sequence
    if l == 1:
        if wvect[0] + rvect[0] < rvect[K]:
            sequence.insert(Operation("Write", [0, 0]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Write_Forward", [0, 2]))
        sequence.insert(Operation("Forward", [1, 2]))
        sequence.insert(Operation("Backward", [2, 1]))
        sequence.insert(Operation("Discard_Forward", [0, 2]))
        if wvect[0] + rvect[0] < rvect[K]:
            sequence.insert(Operation("Read", [0, 0]))
        else:
            sequence.insert(Operation("Read", [K, 0]))
        sequence.insert(Operation("Write_Forward", [0, 1]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Backward", [1, 0]))
        sequence.insert(Operation("Discard_Forward", [0, 1]))
        sequence.insert(Operation("Discard", [0, 0]))
        return sequence
    if K == 0 and cmem == 1:
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(Operation("Read", [0, 0]))
            if index + 1 != 0:
                sequence.insert(Operation("Forward", [0, index + 1]))
            sequence.insert(Operation("Write_Forward", [0, index + 2]))
            sequence.insert(Operation("Forward", [index + 1, index + 2]))
            sequence.insert(Operation("Backward", [index + 2, index + 1]))
            sequence.insert(Operation("Discard_Forward", [0, index + 2]))
        sequence.insert(Operation("Read", [0, 0]))
        sequence.insert(Operation("Write_Forward", [0, 1]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Backward", [1, 0]))
        sequence.insert(Operation("Discard_Forward", [0, 1]))
        sequence.insert(Operation("Discard", [0, 0]))
        return sequence
    if K == 0:
        list_mem = [j * uf + hopt[0][l - j][cmem - 1] + rvect[0] + hoptp[0][j - 1][cmem] for j in range(1, l)]
        if min(list_mem) < hoptp[0][l][1]:
            jmin = argmin(list_mem)
            sequence.insert(Operation("Forward", [0, jmin]))
            sequence.insert_sequence(
                hrevolve_recurse(l - jmin, 0, cmem - 1, cvect, wvect, rvect,
                                 hoptp=hoptp, hopt=hopt, **params).shift(jmin)
            )
            sequence.insert(Operation("Read", [0, 0]))
            sequence.insert_sequence(
                hrevolve_aux(jmin - 1, 0, cmem, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
            )
            aux = sequence
            while aux.type == 'Function':
                aux = aux.sequence[-1]
            if aux.type != "Discard":
                sequence.insert(Operation("Discard", [0, 0]))
            return sequence
        else:
            sequence.insert_sequence(
                hrevolve_aux(l, 0, 1, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params)
            )
            return sequence
    list_mem = [j * uf + hopt[K][l - j][cmem - 1] + rvect[K] + hoptp[K][j - 1][cmem] for j in range(1, l)]
    if min(list_mem) < hopt[K-1][l][cvect[K-1]]:
        jmin = argmin(list_mem)
        sequence.insert(Operation("Forward", [0, jmin]))
        sequence.insert_sequence(
            hrevolve_recurse(l - jmin, K, cmem - 1, cvect, wvect, rvect,
                             hoptp=hoptp, hopt=hopt, **params).shift(jmin)
        )

        sequence.insert(Operation("Read", [K, 0]))
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


def hrevolve(l, cvect, wvect, rvect, fwd_cost, bwd_cost):
    """H-Revolve scheduler.
    
    Parameters
    ----------
    l : int
        The number of forward steps in the initial forward calculation.
    cvect : tuple
        The maximal number of slots that needs to be stored in the levels.
    wvect : tuple
        Number os elements defining the write cost associated with saving a forward 
        restart checkpoint.
    rvect : tuple
        Number os elements defining the read cost associated with copy a forward 
        restart checkpoint from the storage levels.
        _description_, by default None

    Returns
    -------
    Sequence
        The H-Revolve schedules.
    """
    params = revolver_parameters(wvect, rvect, fwd_cost, bwd_cost)
    
    h_rev = hrevolve_recurse(l, len(cvect)-1, cvect[-1], cvect, wvect, rvect,
                            hoptp=None, hopt=None, **params)

    return h_rev


def hrevolve_recurse(l, K, cmem, cvect, wvect, rvect, hoptp=None, hopt=None, **params):
    """Hrevolve recurse schedule.

    Parameters
    ----------
    l : int
        Total number of forward step.
    K : int
        The level of memory.
    cmem : int
        Number of available slots in the K-th level of memory.
        In two level of memory (RAM and Disk), `cmem` collects 
        the number of checkpoints save in Disk.
    cvect : tuple
        The number of slots in each level of memory.
    wvect : tuple
        The cost of writing to each level of memory.
    rvect : tuple
        The cost of reading from each level of memory.
    hoptp : list, optional
        ??, by default None
    hopt : list, optional
        ??, by default None

   
    Returns
    -------
    object
        Hrevolve sequence.

    Raises
    ------
    KeyError
        If `K = 0` and `cmem = 0`.
    """
    parameters = dict(params)
    uf = params["uf"]
    ub = params["ub"]
    if (hoptp is None) or (hopt is None):
        (hoptp, hopt) = get_hopt_table(l, cvect, wvect, rvect, uf, ub)
    sequence = Sequence(Function("HRevolve", l, [K, cmem]),
                        levels=len(cvect), concat=parameters["concat"])
    Operation = partial(Op, params=parameters)
    if l == 0:
        sequence.insert(Operation("Write_Forward", [0, 1]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Backward", [1, 0]))
        sequence.insert(Operation("Discard_Forward", [0, 1]))
        return sequence
    if K == 0 and cmem == 0:
        raise KeyError("It's impossible to execute an AC graph of size > 0 with no memory.")
    if l == 1:
        sequence.insert(Operation("Write", [0, 0]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Write_Forward", [0, 2]))
        sequence.insert(Operation("Forward", [1, 2]))
        sequence.insert(Operation("Backward", [2, 1]))
        sequence.insert(Operation("Discard_Forward", [0, 2]))
        sequence.insert(Operation("Read", [0, 0]))
        sequence.insert(Operation("Write_Forward", [0, 1]))
        sequence.insert(Operation("Forward", [0, 1]))
        sequence.insert(Operation("Backward", [1, 0]))
        sequence.insert(Operation("Discard_Forward", [0, 1]))
        sequence.insert(Operation("Discard", [0, 0]))
        return sequence
    if K == 0:
        sequence.insert(Operation("Write", [0, 0]))
        sequence.insert_sequence(
            hrevolve_aux(l, 0, cmem, cvect, wvect, rvect,
                         hoptp=hoptp, hopt=hopt, **parameters)
        )
        return sequence

    if wvect[K] + hoptp[K][l][cmem] < hopt[K-1][l][cvect[K-1]]:
        sequence.insert(Operation("Write", [K, 0]))
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
