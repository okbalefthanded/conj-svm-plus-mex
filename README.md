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

# Used in
[Iterative Privileged Learning for Multi-view Classification](https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050923X00052/1-s2.0-S1877050923008840/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQC0kkLp3M8lo70wVerJemhIJE7qWNKjitlPuJIhZgt7kQIgUL6DoF4Y4NXdbp8p2871JDaVOpkmM%2B2iMUP7DqB8UJUqswUIdRAFGgwwNTkwMDM1NDY4NjUiDLVOdqoB0fboUWa10CqQBdDz0cWep5jKJfmzVzC5%2BRXjw8j66HDaLF7JZSNmu%2B8SXQibmMp3bQMIcO8T61z3Cxp9KL6fHP%2FlnIjV92wHXi7wZcXIn9cMeKdjgiXo%2B13gpC90CGQq3al2S9FtITzBZW0JSrvCk5GHO%2FhUlxGiv4qS0h0wTYIxSqIpu4YtysXXKQXttORLqqwUpjIjfW5NAAVMRVxb2utN0zs0acSP7UN0vSC2gsP06FOGjJmIvwCq5EDW2wplA5xPdM6Kd5EAA39yk3R%2FLC6nYcnSWacWJz2%2FXg%2FAP2RTQH0nK4GWMhhLUtBKmAqVxch62d31nOAnzl9kFWNXKmhmyPat4rkVGjVZDcXINW2gYpXWQ5R%2BZ8rl4QWACmOMEmt%2FNVIcLtQlIjefRVuTIIxVwG8rrg2qFc30J5pLwDx2FrCdfPtcYdRAYfh97BlfTcHo0nf%2FExmXjlgSvBg0REcs6XNf78DOnqYoE9B%2B3CpgNIBa5XIvQSMP9SXawauONElhtqlEIevvyZAHRk4ubY6ElRRPOrtkOvS4MgfybflJANXJwjzYSdC%2F8GzyJSALH7j50rOPOE0ri3ln4ECZOy8Yp0FoysYNt%2FECm6EkWPdF3tQ6zavr81tihJcpseCz4bHBDrtFil8XPM1g0RKHCP2sIbOSR%2FO4M8Lkv%2BpubRXaeL%2FlYLlMEgSTSU%2Bt4UR73avh13FZZqeg2JtUUSwCIIsOxen4NaLyaQMNOTW29OZKHg%2FRaX2kdOvRNGcSSNCn422Q3q2rlrNqqfk7dDsaqEGfpH5i1bjpz4TcSRzAdrmunzpja6vRn%2BZtHd4Yu%2FDFpLp4yv9fUfSU4pjUqE3P5n4qPNMc3V2rQbu8QKeVjCz97nW8xK5LJdgtMKGY%2BKYGOrEBmbAyJL9HPbCrDcWn8g%2FUAMgbnpyaeOiSmu1NxPzazIZyNWslUbiCoKnfvuN%2FS6j5fByGrBKGJIzrCaKaUyEJlgvqeUYTVe67H%2F2wK650Q%2Bt0Y%2FTXnEGkKIyK%2B1R57dmaHXNrW%2FwWSQ0vuNFDTQMAkHiequkg4EdGGTbkKZorQkslRf%2BUIG596rifIHIDtu80n66EObSxFTXibjfI4wMRFR5FbiLhPT6B7Aykk%2B0o0Ah8&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230817T130409Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSEKAV4OQ%2F20230817%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=b07788b880953943f85e937d642bf358a05064d565bdf5d05f784c0266784e19&hash=a7d4a1a7cceda16e3e0c417477fa95cfc1873ceaf08583fd061945d38002d5fd&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1877050923008840&tid=spdf-e7df7651-e67c-4d4b-9a3c-21ecdf0946af&sid=771c9b4c583d864b5f695586138984f7d291gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=071854075953500855&rr=7f821f09f9b02139&cc=dz)
