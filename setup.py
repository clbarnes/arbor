#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(os.path.join(here, 'catpy', 'version.py')) as f:
    exec(f.read())

with open(os.path.join(here, 'catpy', 'author.py')) as f:
    exec(f.read())

requirements = [
    'numpy>=1.12'
]

setup_requirements = [
    'pytest-runner>=2.11',
]

test_requirements = [
    'pytest>=3',
]

setup(
    name='catpy',
    version=__version__,
    description="Python client for the CATMAID API",
    long_description=readme,
    author=__author__,
    author_email=__email__,
    url='https://github.com/clbarnes/arbor',
    packages=[
        'arbor',
    ],
    package_dir={'arbor':
                 'arbor'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='arbor catmaid neuron',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements
)
