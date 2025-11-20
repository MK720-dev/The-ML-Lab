"""
================================================================================
Name of program: EECS 658 Assignment 1 - Python & ML Libraries Version Check

Brief description: 
This program checks and displays the versions of Python and essential machine 
learning libraries (scikit-learn, NumPy, Pandas, SciPy) to verify proper 
installation and environment setup. It also prints "Hello World!" as a basic 
functionality test.

Inputs:
- None (program retrieves version information from installed packages)

Output:
- Prints out the version of Python
- Prints out the version of scikit-learn
- Prints out the version of NumPy  
- Prints out the version of Pandas
- Prints out the version of SciPy
- Prints out "Hello World!"

Collaborators: None

Other sources: None

Author: Malek Kchaou

Creation date: August 28, 2025
================================================================================
"""

# Python version
import sys
print("Python version:", sys.version)

# scikit-learn version
import sklearn
print("scikit-learn version:", sklearn.__version__)

# NumPy version
import numpy as np
print("NumPy version:", np.__version__)

# Pandas version
import pandas as pd
print("Pandas version:", pd.__version__)

# SciPy version
import scipy
print("SciPy version:", scipy.__version__)

#Hello World
print("Hello World!")
