from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _read_version():
    version_ns = {}
    with open(path.join(here, "selkit", "version.py"), encoding="utf-8") as f:
        exec(f.read(), version_ns)
    return version_ns["__version__"]


CLASSIFIERS = [
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "License :: OSI Approved :: MIT License",
]

REQUIRES = [
    "numpy>=1.26",
    "scipy>=1.12",
    "rich>=13.7",
    "PyYAML>=6.0",
]

setup(
    name="selkit",
    description="Python reimplementation of PAML's selection-analysis workflows (codeml, yn00).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob L. Steenwyk",
    author_email="jlsteenwyk@gmail.com",
    url="https://github.com/JLSteenwyk/selkit",
    packages=find_packages(include=["selkit*"]),
    classifiers=CLASSIFIERS,
    entry_points={"console_scripts": ["selkit = selkit.__main__:main"]},
    version=_read_version(),
    include_package_data=True,
    install_requires=REQUIRES,
    python_requires=">=3.11",
)

## push new version to pypi
# rm -rf dist
# python3 setup.py sdist bdist_wheel --universal
# twine upload dist/* -r pypi
