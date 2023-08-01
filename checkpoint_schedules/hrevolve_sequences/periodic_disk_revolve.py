
"""This module contains the periodic disk revolve checkpoint schedule."""
from functools import partial
from .basic_functions import (Operation as Op, Sequence, Function, beta)
from .revolve import revolve, get_opt_0_table
from .revolve_1d import get_opt_1d_table, revolve_1d
from .utils import revolver_parameters


def compute_mmax(params):
    """Compute the maximum period.

    Parameters
    ----------
    cm : int
        The number of checkpoints stored in memory.
    wd : float
        Cost of writing to disk.
    rd : float
        Cost of reading from disk.
    uf : int
        Cost of the forward steps.

    Returns
    -------
    _type_
        _description_
    """
    cm = params["cm"]
    wd = params["wd"]
    rd = params["rd"]
    uf = params["uf"]
    td1 = 0
    while beta(cm, td1) <= (wd + rd) / uf:
        td1 += 1
    td2 = 0
    while beta(cm, td2) <= wd / uf:
        td2 += 1
    return int(max(beta(cm, td1+1), 2*beta(cm, td2) + 1))


def rel_cost_x(m, opt_1d_m_moins_1, wd, rd):
    """ The RelCost_X function

    Parameters
    ----------
    m : int
        The period.
    opt_1d_m_moins_1 : float
        The cost of the optimal 1D schedule for period m-1.
    wd : float
        Cost of writing to disk.
    rd : float
        Cost of reading from disk.
    """
    return 1.0*(wd + rd + opt_1d_m_moins_1) / m


def compute_mx(cm, opt_0=None, opt_1d=None, mmax=None, **params):
    """Compute the optimal period.

    Parameters
    ----------
    cm : _type_
        _description_
    opt_0 : _type_, optional
        _description_, by default None
    opt_1d : _type_, optional
        _description_, by default None
    mmax : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if mmax is None:
        mmax = compute_mmax(params)
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
    """Compute mX using the close formula in the paper
        It's not proven yet, but it's been verified and it's faster

    Parameters
    ----------
    cm : _type_
        _description_
    rd : _type_
        _description_
    wd : _type_
        _description_
    opt_0 : _type_, optional
        _description_, by default None
    opt_1d : _type_, optional
        _description_, by default None
    """
    def f(x, y, c):
        return int(beta(c + 1, x + y - 1) - sum([beta(c, k) for k in range(0, y)]))

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
    if (rel_cost_x(mx, opt_1d[mx-1], wd, rd) < rel_cost_x(mxalt, opt_1d[mxalt-1], wd, rd)):
        return int(mx)
    else:
        return int(mxalt)


def mxrr_close_formula(cm, uf, rd, wd):
    """Compute mXrr using the close formula in the paper.

    Parameters
    ----------
    cm : int
        The number of checkpoints stored in memory.
    uf : float
        Cost of the forward step.
    rd : float
        Cost of reading from disk.
    wd : float
        Cost of writing to disk.

    Returns
    -------
    int
        
    """
    t = 0
    while beta(cm+1, t) <= (wd + rd) / uf:
        t += 1
    return int(beta(cm, t))


def periodic_disk_revolve(l, cm, rd, wd, fwd_cost, bwd_cost, opt_0=None, 
                          opt_1d=None, mmax=None):
    """Periodic disk revolve algorithm.
            
    Parameters
    ----------
    l : int
        The number of forward step to execute in the AC graph.
    cm : int
        The number of slots available in memory.
    rd : float
        Cost of read the checkpoint data from disk.
    wd : float
        Cost of writing the checkpoint data to disk.
    fwd_cost : float
        Cost of the forward step.
    bwd_cost : float
        Cost of the backward step.
    opt_0 : _type_, optional
        _description_, by default None
    opt_1d : _type_, optional
        _description_, by default None
    mmax : int, optional
        The maximum period to consider, by default None.

    Returns
    -------
    Sequence
        Return the periodic disk revolve schedule.
    """
    
    params = revolver_parameters(wd, rd, fwd_cost, bwd_cost)
    parameters = dict(params)
    # parameters.update(params)
    mx = parameters["mx"]
    one_read_disk = parameters["one_read_disk"]
    fast = parameters["fast"]
    if mmax is None:
        if one_read_disk:
            mmax = mxrr_close_formula(cm, fwd_cost, rd, wd)
            if mx is None:
                mx = mmax
        else:
            mmax = compute_mmax(parameters)
    if mx is not None:
        mmax = max(mmax, mx) + 1
    if opt_0 is None:
        opt_0 = get_opt_0_table(mmax, cm, **parameters)
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(mmax, cm, opt_0=opt_0, **parameters)
    sequence = Sequence(Function("Periodic-Disk-Revolve", l, cm),
                        concat=parameters["concat"])
    operation = partial(Op, params=parameters)
    if mx is None:
        if one_read_disk:
            mx = mxrr_close_formula(cm, fwd_cost, rd, wd)
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
    if one_read_disk or opt_1d[l - current_task] == opt_0[cm][l - current_task]:
        sequence.insert_sequence(
            revolve(l - current_task, cm, rd, wd, fwd_cost, bwd_cost,
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
                revolve(mx - 1, cm, rd, wd, fwd_cost, bwd_cost,
                        opt_0=opt_0).shift(current_task)
            )
        else:
            sequence.insert_sequence(
                revolve_1d(mx-1, cm, opt_0=opt_0, opt_1d=opt_1d, 
                           **parameters).shift(current_task)
            )
    return sequence
