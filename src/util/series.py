import numpy as np
import sys

def infsum(f, eps=1.e-6, n0 = 0, n_min=5, n_limit=10000, twosides = True):
    res0=f(n0)
    if twosides:
        res=f(n0)+f(n0+1)+f(n0-1)
    else:
        res=f(n0)+f(n0+1)
    n=2
    while True:
        if n>n_limit:
            sys.exit('infsum: n exceeds n_limit')
        if np.abs(res-res0) < eps*np.abs(res+res0)/2 and n >= n_min:
            break
        res0=res
        if twosides:
            res += f(n0+n) + f(n0-n)
        else:
            res += f(n0+n)
        n += 1
    return res