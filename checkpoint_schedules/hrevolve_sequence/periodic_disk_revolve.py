from .parameters import defaults
from .basic_functions import (Operation as Op, Sequence, Function, beta)
from .revolve import revolve, get_opt_0_table
import math
from functools import partial

from .revolve_1d import get_opt_1d_table, revolve_1d


def compute_mmax(cm, wd, rd, uf, **params):
    """ Compute m_max, the bound on the optimal
        period mX, as defined in the paper """
    td1 = 0
    while beta(cm, td1) <= (wd + rd) / uf:
        td1 += 1
    td2 = 0
    while beta(cm, td2) <= wd / uf:
        td2 += 1
    return int(max(beta(cm, td1+1), 2*beta(cm, td2) + 1))


def RelCostX(m, opt_1d_m_moins_1, wd, rd, **params):
    """ The RelCost_X function, as defined in the paper """
    return 1.0*(wd + rd + opt_1d_m_moins_1) / m


def compute_mx(cm, opt_0=None, opt_1d=None, mmax=None, **params):
    """ Compute the optimal period mX, as defined in the paper """
    if mmax is None:
        mmax = compute_mmax(cm, **params)
    if opt_0 is None or len(opt_0) < mmax:
        opt_0 = get_opt_0_table(mmax, cm, **params)
    if opt_1d is None or len(opt_1d) < mmax:
        opt_1d = get_opt_1d_table(mmax, cm, opt_0=opt_0, **params)
    mx = 1
    objbest = RelCostX(1, opt_1d[0], **params)
    for mxi in range(2, mmax+1):
        obj = RelCostX(mxi, opt_1d[mxi-1], **params)
        # print("mxi", mxi, "opt", opt_1d[mxi-1], "wd", wd, "rd", rd, "obj", obj)
        if obj <= objbest:
            objbest = obj
            mx = mxi
    return mx


def mx_close_formula(cm, rd, wd, opt_0=None, opt_1d=None, **params):
    """ Compute mX using the close formula in the paper
        It's not proven yet, but it's been verified and it's faster """
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
    if (RelCostX(mx, opt_1d[mx-1]) < RelCostX(mxalt, opt_1d[mxalt-1])):
        return int(mx)
    else:
        return int(mxalt)


def mxrr_close_formula(cm, uf, rd, wd, **params):
    """ Compute mXrr using the close formula in the paper"""
    t = 0
    while beta(cm+1, t) <= (wd + rd) / uf:
        t += 1
    # t -= 1
    return int(beta(cm, t))


def combin(k, n):
    return int(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))


def periodic_disk_revolve(l, cm, opt_0=None, opt_1d=None, mmax=None, **params):
    """ l : number of forward step to execute in the AC graph
            cm : number of available memory slots
            Return the periodic sequence with optimal period"""
    parameters = dict(defaults)
    parameters.update(params)
    mx = parameters["mx"]
    one_read_disk = parameters["one_read_disk"]
    fast = parameters["fast"]
    if mmax is None:
        if one_read_disk:
            mmax = mxrr_close_formula(cm, **parameters)
            if mx is None:
                mx = mmax
        else:
            mmax = compute_mmax(cm, **parameters)
    if mx is not None:
        mmax = max(mmax, mx) + 1
    if opt_0 is None:
        opt_0 = get_opt_0_table(mmax, cm, **parameters)
    if opt_1d is None and not one_read_disk:
        opt_1d = get_opt_1d_table(mmax, cm, opt_0=opt_0, **parameters)
    sequence = Sequence(Function("Periodic-Disk-Revolve", l, cm), concat=parameters["concat"])
    Operation = partial(Op, params=parameters)
    if mx is None:
        if one_read_disk:
            mx = mxrr_close_formula(cm, **parameters)
        elif fast:
            mx = mx_close_formula(cm, opt_0=opt_0, opt_1d=opt_1d, **parameters)
        else:
            mx = compute_mx(cm, opt_0=opt_0, opt_1d=opt_1d, **parameters)
    print("We use periods of size ", mx)
    current_task = 0
    while l - current_task > mx:
        sequence.insert(Operation("Write_disk", current_task))
        sequence.insert(Operation("Forward", [current_task, current_task + mx - 1]))
        current_task += mx
    if one_read_disk or opt_1d[l - current_task] == opt_0[cm][l - current_task]:
        sequence.insert_sequence(
            revolve(l - current_task, cm, opt_0=opt_0, **parameters).shift(current_task)
        )
    else:
        sequence.insert(Operation("Write_disk", current_task))
        sequence.insert_sequence(
            revolve_1d(l - current_task, cm, opt_0=opt_0, opt_1d=opt_1d, **parameters).shift(current_task)
        )
    while current_task > 0:
        current_task -= mx
        sequence.insert(Operation("Read_disk", current_task))
        if one_read_disk:
            sequence.insert_sequence(
                revolve(mx - 1, cm, opt_0=opt_0, **parameters).shift(current_task)
            )
        else:
            sequence.insert_sequence(
                revolve_1d(mx-1, cm, opt_0=opt_0, opt_1d=opt_1d, **parameters).shift(current_task)
            )
    return sequence
