#!/usr/bin/env python

from setuptools import setup, find_packages

import icb

setup(
    name='icb',
    version=icb.__version__,
    description='Reimplementation of approaches for issue classification',
    install_requires=['scikit-learn', 'pandas','skift','numpy','nltk', 'keras', 'pycoshark', 'tensorflow',
    'skfeature @ http://github.com/smartshark/scikit-feature/tarball/master#egg=skfeature-1.0.0'],
    dependency_links=['http://github.com/smartshark/scikit-feature/tarball/master#egg=skfeature-1.0.0'],
    author='ftrautsch',
    author_email='fabian.trautsch@informatik.uni-goettingen.de',
    url='https://github.com/smartshark/icb',
    download_url='https://github.com/smartshark/icb/zipball/master',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
