import os
import sys
import setuptools 

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 10)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
DeepDive requires Python %s.%s or higher, but you're trying to
install it on Python %s.%s.
""" % (REQUIRED_PYTHON[0], REQUIRED_PYTHON[1], CURRENT_PYTHON[0], CURRENT_PYTHON[1]))
    sys.exit(1)

requirements_list = [
     "numpy>=1.22.3",
     "matplotlib>=3.5.2",
     "pandas>=1.4.3",
     "scipy>=1.8.1",
     "tensorflow>=2.8.0",
     "seaborn>=0.11.2",
     "keras>=2.8.0",   
    ]

setuptools.setup(
    name="DeepDive",
    version="1.17c",
    author="Daniele Silvestro",
    description="Diversity estimation using deep learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=requirements_list,
)




