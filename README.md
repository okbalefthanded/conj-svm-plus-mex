# conj-svm-plus-mex
A mex interface for SMO style SVM+ solvers developed on top of LIBSVM package In [1] and [2], the code follows the same structure of the original LIBSVM mex interface.
The original code developed by Dmitry Pechyony is also provided in the _orignal code_ folder.

# Installation

Before running the make.m script, check for a valid C/C++ compiler installed in your system; once you have passed this step just execute the make.m script and it will build the mex functions.

* 64-bits windows compiled mex functions are provided in this repo, Microsoft Visual C++ 2012 was used as a compiler (Microsoft Visual C++ 2013 was tested succussfully).
* The code was tested on Matlab R2014a and R2015a.

# Usage 

Run the svm_plus_demo.m script

# Limitations

Please refer to the issues section

# Cite us
Coming soon

# References
[1] D. Pechyony, R. Izmailov, A. Vashist and V. Vapnik. SMO-style Algorithms for Learning using Privileged Information . DMIN 2010.

[2] D. Pechyony and V. Vapnik. Fast Optimization Algorithms for Solving SVM+. Chapter in Statistical Learning and Data Science, 2011.
