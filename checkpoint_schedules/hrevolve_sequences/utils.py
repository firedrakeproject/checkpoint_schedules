#!/usr/bin/python
# Copyright (C) 2020 - 2024 Inria and Imperial College London
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version. This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.
# Developed originally by Guillaume Pallez (guillaume.pallez@inria.fr), Julien Herrmann (jln.herrmann@gmail.com).
# Modified by Daiane I. Dolci (d.dolci@eimperial.ic.ac.uk) and David A. Ham (david.ham@imperial.ac.uk).

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
