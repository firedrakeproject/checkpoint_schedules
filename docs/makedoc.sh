rm -Rf build
sphinx-apidoc -f -o docs/source/docstring/ ../checkpoint_schedules
make html