# checkpoint_schedules
*checkpoint_schedules* provides schedules for step-based incremental
checkpointing of the adjoints to computer models. The schedules contain
instructions indicating the sequence of forward and adjoint steps to be
executed, and the data storage and retrieval to be performed. 

The schedule instructions are independent of the model implementation,
which enables the model authors to switch between checkpointing algorithms
without recoding their adjoint solvers. Conversely, *checkpoint_schedules*
provides developers of checkpointing algorithms with a direct mechanism to convey
their work to model authors.

## Installation
*checkpoint_schedules* is a Python package and can be installed using pip
```
pip install checkpoint-schedules
```

## Usage
The usage guide of *checkpoint_schedules* is available [here](https://nbviewer.org/github/firedrakeproject/checkpoint_schedules/blob/main/docs/notebooks/tutorial.ipynb).


## Contributing
We welcome contributions to *checkpoint_schedules*!
To contribute please consider the following steps:
1. Fork the repository.
2. Make your changes.
3. Make sure that the tests pass by running `pytest test` and `pytest --nbval-lax docs/notebooks/`
4. Add tests for your changes (if applicable).
5. Add documentation for your changes that follows the [Sphinx](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) format.
6. Submit your pull request.

## Bug reports
Please report bugs on the [issue tracker](https://github.com/firedrakeproject/checkpoint_schedules/issues).

## Documentation
The complete documentation for checkpoint_schedules is available at [Firedrake project website](https://www.firedrakeproject.org/checkpoint_schedules/).

## License
*checkpoint_schedules* is licensed under the GNU LGPL version 3. See the LICENSE file for details.
