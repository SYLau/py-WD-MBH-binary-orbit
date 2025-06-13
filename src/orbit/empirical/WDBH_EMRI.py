import numpy as np
if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G, msun, rsun

def rWD(mWD):
    mCh = 1.44*msun
    return 9.04e8*(mWD/mCh)**(-1./3)*(1-mWD/mCh)**0.447
# def rWD(mWD):
#     mCh = 1.44*msun
#     mJ = 0.00057*msun
#     x = mWD/mCh
#     y = mWD/mJ
#     return 0.0114*rsun*np.sqrt(x**(-2./3) - x**(2./3))*(1+3.5*y**(-2./3) + y**(-1))**(-2./3)

def rWD_E86(mWD):
    mCh = 1.44*msun
    mJ = 0.00057*msun
    x = mWD/mCh
    y = mWD/mJ
    return 0.0114*rsun*np.sqrt(x**(-2./3) - x**(2./3))*(1+3.5*y**(-2./3) + y**(-1))**(-2./3)

def rtide(mWD,mBH):
    return rWD(mWD)*(mBH/mWD)**(1./3)

def wf(mWD):
    return 1.455*np.sqrt(G*mWD/rWD(mWD)**3)

def olapf(mWD,Qnl):
    return np.sqrt(G*mWD/2/rWD(mWD)**3/wf(mWD)**2)*Qnl

def rRL(mWD,mBH,rp):
    q = mWD/mBH
    return rp*(0.49*q**(2./3))/(0.6*q**(2./3)+np.log(1+q**(1./3)))

def tutorial():
    mWD = 0.5*msun
    mBH = 1e5*msun
    rp = 2.*rtide(mWD,mBH)

    print('mWD = ', mWD/msun)
    print('rWD = ', rWD(mWD)/1.e5)
    print('R_tide = ', rtide(mWD,mBH))
    print('w_f = ', wf(mWD))
    print('olap_f = ', olapf(mWD,0.5))
    print('r_peri = ', rp)
    print('Roche lobe = ', rRL(mWD,mBH,rp))

    return

if __name__ == '__main__':
    tutorial()