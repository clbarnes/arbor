# arbor.py

[![Build Status](https://travis-ci.org/clbarnes/arbor.svg?branch=master)](https://travis-ci.org/clbarnes/arbor)

An implementation of 
[Arbor.js](https://github.com/catmaid/CATMAID/blob/master/django/applications/catmaid/static/libs/catmaid/Arbor.js)
and related tools, for Python 3.7+
 
N.B. This is a work in progress, but isn't really being worked on (by me, anyway):
it will eventually be replaced by bindings to a rust implementation which could also be used in the frontend as WASM.
Pull requests are welcome, however!

# Contributing

1. Development install with `make install-dev`
2. Format code with `make fmt` (uses `black`)
3. Lint with `make lint` (uses `pyflakes`) 
4. Test with `make test` (uses `pytest`)
  - This also downloads the reference JS implementation, runs them on a real arbor, and dumps them out for reference
  
Run `hooks/install.sh` (unix-only) to install a pre-commit hook which formats changed python files
and runs the linter.

## Notes

Tests of functions which run very slowly are skipped by default; 
use `pytest --skipslow` to skip them, 
and `@pytest.mark.slow` to mark new tests as slow.
