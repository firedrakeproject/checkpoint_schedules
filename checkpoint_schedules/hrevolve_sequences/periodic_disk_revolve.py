# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr),
# Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk)
# and David A. Ham (david.ham@imperial.ac.uk).

"""This module contains the periodic disk revolve checkpoint schedule."""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, beta)
from .revolve import revolve, get_opt_0_table
from .revolve_1d import get_opt_1d_table, revolve_1d
from .utils import revolver_parameters


def compute_mmax(cm, wd, rd, uf):
    """Compute the maximum period.

    Parameters
    ----------
    cm : int
        Memory slots.
    wd : float
        Cost of writing the checkpoint data in disk.
    rd : float
        Cost of reading the checkpoint data from disk.
    uf : float
        The cost of advancing the forward over one step.

    Returns
    -------
    int
        The maximum period.
    """
    td1 = 0
    while beta(cm, td1) <= (wd + rd) / uf:
        td1 += 1
    td2 = 0
    while beta(cm, td2) <= wd / uf:
        td2 += 1
    return int(max(beta(cm, td1 + 1), 2 * beta(cm, td2) + 1))


def rel_cost_x(m, opt_1d_m_moins_1, wd, rd):
    """Compute the relative cost of the optimal execution time.

    Parameters
    ----------
    m : int
        The period.
    opt_1d_m_moins_1 : list
        The optimal execution time.
    wd : float
        Cost of writing the checkpoint data in disk.
    rd : float
        Cost of reading the checkpoint data from disk.

    Returns
    -------
    float
        The relative cost of the optimal execution time.
    """
    return 1.0*(wd + rd + opt_1d_m_moins_1) / m


def compute_mx(cm, opt_0=None, opt_1d=None, mmax=None, **params):
    """Compute the period.

    Parameters
    ----------
    cm : int
        Memory slots.
    opt_0 : list, optional
        Optimal execution time for a memory revolve algorithm.
    opt_1d : list, optional
        Optimal execution time for a 1D revolve algorithm.
    mmax : int, optional
        The maximum period.
    params : dict
        The parameters dictionary.

    Notes
    -----
    This period only depends of `cm`, `wd` and `rd`.
    In the current case, the forward computations performed
    between two consecutive checkpoints into the second level
    of storage is always the same except for a bounded number of them.
    Additional details about the period computation, refers to the paper [1].

    [1] Aupy, G.,  Herrmann, J. "Periodicity in optimal hierarchical
    checkpointing schemes for adjoint computations". Optimization Methods and
    Software, 32(3), 594-624, (2017).
    DOI: 10.1080/10556788.2016.1230612.

    Returns
    -------
    int
        The period.
    """
    if mmax is None:
        mmax = compute_mmax(params["cm"], params["wd"], params["rd"],
                            params["uf"])
    if opt_0 is None or len(opt_0) < mmax:
        opt_0 = get_opt_0_table(mmax, cm, **params)
    if opt_1d is None or len(opt_1d) < mmax:
        opt_1d = get_opt_1d_table(mmax, cm, opt_0=opt_0, **params)
    mx = 1
    objbest = rel_cost_x(1, opt_1d[0], params["wd"], params["rd"])
    for mxi in range(2, mmax+1):
        obj = rel_cost_x(mxi, opt_1d[mxi-1], params["wd"], params["rd"])
        if obj <= objbest:
            objbest = obj
            mx = mxi
    return mx


def mx_close_formula(cm, rd, wd, opt_0=None, opt_1d=None, **params):
    """Compute the period by a closed formula.

    Parameters
    ----------
    cm : int
        Memory slots.
    rd : float
        Cost of reading the checkpoint data from disk.
    wd : float
        Cost of writing the checkpoint data in disk.
    opt_0 : list, optional
        Optimal execution time for a memory revolve algorithm.
    opt_1d : list, optional
        Optimal execution time for a 1D revolve algorithm.
    params : dict
        The parameters dictionary.

    Notes
    -----
    This period is computed by a closed formula. The authors of the paper [1]
    showed that a solution that checkpoints periodically data to disks is
    asymptotically optimal, both in the offline case (the number of steps is
    known before-hand), and in the online case (the number of steps is not
    known before-hand).

    [1] Aupy, G.,  Herrmann, J. "Periodicity in optimal hierarchical
    checkpointing schemes for adjoint computations". Optimization Methods and
    Software, 32(3), 594-624, (2017).
    DOI: 10.1080/10556788.2016.1230612.
    """
    def f(x, y, c):
        return int(beta(c + 1, x + y - 1) - sum([beta(c, k)
                                                 for k in range(0, y)]))

    x = 0
    while (rd >= beta(cm+1, x)):
        x += 1
    y = 0
    while (wd > sum([f(j, x, cm) for j in range(1, y+1)])):
        y += 1
    mx = f(y, x, cm)
    x += 1
    y = 0
    while (wd > sum([f(j, x, cm) for j in range(1, y+1)])):
        y += 1
    mxalt = f(y, x, cm)
    mmax = max(mx, mxalt)
    if opt_0 is None or len(opt_0) < mmax:
        opt_0 = get_opt_0_table(mmax, cm, **params)
    if opt_1d is None or len(opt_1d) < mmax:
        opt_1d = get_opt_1d_table(mmax, cm, opt_0=opt_0, **params)
    if (rel_cost_x(mx, opt_1d[mx-1], wd, rd) < rel_cost_x(mxalt,
                                                          opt_1d[mxalt-1], wd,
                                                          rd)):
        return int(mx)
    else:
        return int(mxalt)


def mxrr_close_formula(cm, uf, rd, wd):
    """Compute the period that minimises asymptotically
    the execution time of the periodic disk revolve algorithm.

    Parameters
    ----------
    cm : int
        Memory slots.
    uf : float
        The cost of advancing the forward over one step.
    rd : float
        Cost of reading the checkpoint data from disk.
    wd : float
        Cost of writing the checkpoint data in disk.

    Returns
    -------
    int
        The period.
    """
    t = 0
    while beta(cm+1, t) <= (wd + rd) / uf:
        t += 1
    return int(beta(cm, t))


def periodic_disk_revolve(l, cm, rd, wd, uf, ub, opt_0=None,  # noqa: E741
                          opt_1d=None, mmax=None):
    """Periodic disk revolve algorithm.

    Parameters
    ----------
    l : int
        The number of forward step to execute in the AC graph.
    cm : int
        Memory slots.
    rd : float
        Cost of read the checkpoint data from disk.
    wd : float
        Cost of writing the checkpoint data to disk.
    ub : float
        The cost of advancing the adjoint over one step.
    uf : float
        The cost of advancing the forward over one step.
    opt_0 : _type_, optional
        _description_, by default None
    opt_1d : _type_, optional
        _description_, by default None
    mmax : int, optional
        The maximum period to consider, by default None.

    Notes
    -----
    Memory slots are also interpreted as the number of checkpoints stored in
    memory.

    Returns
    -------
    Sequence
        Return the periodic disk revolve schedule.
    """
    params = revolver_parameters(wd, rd, uf, ub)
    parameters = dict(params)
    # parameters.update(params)
    mx = parameters["mx"]
    one_read_disk = parameters["one_read_disk"]
    fast = parameters["fast"]
    if mmax is None:
        if one_read_disk:
            mmax = mxrr_close_formula(cm, uf, rd, wd)
            if mx is None:
                mx = mmax
        else:
            mmax = compute_mmax(params["cm"], params["wd"], params["rd"],
                                params["uf"])
    if mx is not None:
        mmax = max(mmax, mx) + 1
    if opt_0 is None:
        opt_0 = get_opt_0_table(mmax, cm, params["uf"], params["ub"])
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(mmax, cm, ub, uf, rd, one_read_disk,
                                  opt_0=opt_0)
    sequence = Sequence(Function("Periodic-Disk-Revolve", l, cm),
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if mx is None:
        if one_read_disk:
            mx = mxrr_close_formula(cm, uf, rd, wd)
        elif fast:
            mx = mx_close_formula(cm, opt_0=opt_0, opt_1d=opt_1d, **parameters)
        else:
            mx = compute_mx(cm, opt_0=opt_0, opt_1d=opt_1d, **parameters)
    print("We use periods of size ", mx)
    current_task = 0
    while l - current_task > mx:
        sequence.insert(operation("Write_disk", current_task))
        sequence.insert(operation("Forward",
                                  [current_task, current_task + mx]))
        current_task += mx
    if one_read_disk or opt_1d[l - current_task] == opt_0[cm][l - current_task]:  # noqa: E501
        sequence.insert_sequence(
            revolve(l - current_task, cm, rd, wd, uf, ub,
                    opt_0=opt_0).shift(current_task)
        )
    else:
        sequence.insert(operation("Write_disk", current_task))
        sequence.insert_sequence(
            revolve_1d(l - current_task, cm, opt_0=opt_0, opt_1d=opt_1d,
                       **parameters).shift(current_task)
        )
    while current_task > 0:
        current_task -= mx
        sequence.insert(operation("Read_disk", current_task))
        if one_read_disk:
            sequence.insert_sequence(
                revolve(mx - 1, cm, rd, wd, uf, ub,
                        opt_0=opt_0).shift(current_task)
            )
        else:
            sequence.insert_sequence(
                revolve_1d(mx-1, cm, opt_0=opt_0, opt_1d=opt_1d,
                           **parameters).shift(current_task)
            )
    return sequence
