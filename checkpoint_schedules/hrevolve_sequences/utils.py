#!/usr/bin/python

def revolver_parameters(wd, rd, uf, ub):
    """Parameter use to obtain the revolver sequences.

    Parameters
    ----------
    wd : float
        Cost of writing to disk.
    rd : float
        Cost of reading from disk.
    uf : float
        Measure of the forward cost related to one time-step execution.
    ub : float
        Measure of the backward cost related to one time-step execution.

    Returns
    -------
    dict
        Dictionary of parameters used in the revolver algorithm.
    """

    params = {
        "uf": uf,  # Cost of a forward step.
        "ub": ub,  # Cost of a backward step.
        "up": 1,   # Cost of the loss function.
        "wd": wd,  # Cost of writing to disk.
        "rd": rd,  # Cost of reading from disk.
        "mx": None,  # Size of the period (defaults to the optimal).
        "one_read_disk": True,  # Disk checkpoints are only read once.
        "fast": False,  # Use the clode formula for mx.
        "concat": 0,  # Level of sequence concatenation.
        "print_table": "None",  # File to which to print the results table.
    }
    return params
