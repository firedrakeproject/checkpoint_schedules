.. _checkpoint_schedules-documentation:
.. title:: checkpoint_schedules documentation

********************
checkpoint_schedules
********************
:Authors:       Daiane I. Dolci, James R. Maddison, David A. Ham, Guillaume Pallez, Julien Herrmann
:Contact:      d.dolci@imperial.ac.uk
:GitHub:       https://github.com/firedrakeproject/checkpoint_schedules
:Date:         |today|

Quickstart
==========

If you want to quickly get up and running with *checkpoint_schedules* your adjoint problem, follow these steps:
 
* Install *checkpoint_schedules* via pip ::

  $  pip install checkpoint-schedules

* Familiarise with *checkpoint_schedules* by acessing the :ref:`introduction <introduction>`.
* Go through to an `illustrative example <https://nbviewer.org/github/firedrakeproject/checkpoint_schedules/blob/main/docs/notebooks/tutorial.ipynb>`_ that explains the usage of *checkpoint_schedules* for step-based incremental checkpointing of the adjoints.
* See `here <https://nbviewer.org/github/firedrakeproject/checkpoint_schedules/blob/main/docs/notebooks/burger.ipynb>`_ for an adjoint problem solved using *checkpoint_schedules*.

API documentation
=================

The complete list of all the classes and methods in *checkpoint_schedules* is available at the :ref:`API reference
<checkpoint_schedules-api-reference>`.


Contributing
============
We welcome contributions to *checkpoint_schedules*!
To contribute please consider the following steps:

1. Fork the repository.

2. Make your changes.

3. Make sure that the tests pass by running `pytest test` and `pytest --nbval-lax docs/notebooks/`.

4. Add tests for your changes (if applicable).

5. Add documentation for your changes that follows the `Sphinx <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_ format.

6. Submit a pull request.

Bug reports
===========
Please report bugs on the `issue tracker <https://github.com/firedrakeproject/checkpoint_schedules/issues>`_.

Citing
======
If you use *checkpoint_schedules* in your research, please cite `JOSS paper <https://joss.theoj.org/papers/10.21105/joss.06148>`_.