import os
import sys
import setuptools 

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)

if CURRENT_PYTHON < REQUIRED_PYTHON or CURRENT_PYTHON > MAX_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
DeepDive requires Python versions between 3.%s and 3.%s, but you're trying to
install it on Python %s.%s.
""" % ( REQUIRED_PYTHON[1], MAX_PYTHON[1], CURRENT_PYTHON[0], CURRENT_PYTHON[1]))
    sys.exit(1)


requirements_list = [
     "numpy>=1.22.3",
     "matplotlib>=3.5.2",
     "pandas>=1.4.3",
     "scipy>=1.8.1",
     "tensorflow>=2.8.0",
     "seaborn>=0.11.2",
     "keras>=2.8.0",
     "scikit-learn>=1.6.1",
    ]

setuptools.setup(
    name="DeepDive",
    version="1.0.25",
    author="Daniele Silvestro",
    description="Diversity estimation using deep learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=requirements_list,
)




