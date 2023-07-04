.. _install:

*************
Installation
*************

Requirements
-------------
The supported Python versions are:

- `Python 3.7`
- `Python 3.8`
- `Python 3.9`
- `Python 3.10`
- `Python 3.11`

Installation via pip
--------------------

After installing all dependencies, perform ::

$  pip install checkpoint_schedule

.. note ::
    If you do not have administrative rights on your system, add the flag ``--user`` to the command above.

Developers
^^^^^^^^^^

For developers using `pip`, simply invoke the `-e` flag in the installation command in your local clone ::

    git clone https://github.com/firedrakeproject/checkpoint_schedules.git
    cd checkpoint_schedules
    pip install -e .

The last line makes sure that all your code changes are included whenever importing *checkpoint_schedule*. 
Concerning the `git clone` line, we actually recommend that you fork `checkpoint_schedule` on GitHub and then replace that command by cloning your fork instead.
