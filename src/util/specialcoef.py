import numpy as np

def fact(r):
    n= int(r)
    if n < 2:
        return 1
    else:
        return n * fact(n-1)

def Wlm(ell,m):
    lm_p=ell+m
    lm_m=ell-m
    if int(lm_p)%2 != 0:
        return 0.
    else:
        return (-1.)**(lm_p/2)*np.sqrt(4*np.pi/(2*ell+1)*fact(lm_p)* fact(lm_m)) \
            /(2**ell*fact(lm_p/2)*fact(lm_m/2))