# arbor.py

A fairly honest reimplementation of 
[Arbor.js](https://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/libs/catmaid/Arbor.js)
 and related tools, for Python 3.7+
 
# Notes

Tests of functions which run very slowly are skipped by default; use `pytest --runslow`

# Contributing

1. Development install with `make install-dev`
2. Format code with `make fmt` (uses `black`)
3. Lint with `make lint` (uses `pyflakes`) 
4. Test with `make test` (uses `pytest`)
  - This also downloads the reference JS implementation, runs them on a real arbor, and dumps them out for reference
