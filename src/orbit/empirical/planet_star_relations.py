import numpy as np
if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G, msun, rsun, mearth, rearth

def rPL(mPL):
    mPL_arr = np.asarray(mPL)
    x = np.log10(mPL_arr / mearth)
    y = np.where(
        (mPL_arr / mearth) >= 20,
        -0.01043 * x**3 - 0.002476 * x**2 + 0.3036 * x + 0.4603,
        -0.02696 * x**2 + 0.2827 * x + 0.04447
    )
    result = 10**y * rearth
    # Return scalar if input was scalar
    if np.isscalar(mPL):
        return result.item()
    else:
        return result

def rtide(mPL,mST):
    return rPL(mPL)*(mST/mPL)**(1./3)

def wf(mPL):
    return 1.3*np.sqrt(G*mPL/rPL(mPL)**3)

def olapf(mPL,Qnl):
    return np.sqrt(G*mPL/2/rPL(mPL)**3)/wf(mPL)*Qnl

def rRL(mPL,mST,rp):
    q = mPL/mST
    return rp*(0.49*q**(2./3))/(0.6*q**(2./3)+np.log(1+q**(1./3)))

def tutorial():
    mPL = 50*mearth
    mST = 1.*msun
    rp = 2.5*rtide(mPL,mST)

    print('mPL = ', mPL/msun)
    print('rWD = ', rPL(mPL)/1.e5)
    print('R_tide = ', rtide(mPL,mST))
    print('w_f = ', wf(mPL))
    print('olap_f = ', olapf(mPL,0.5))
    print('r_peri = ', rp)
    print('Roche lobe = ', rRL(mPL,mST,rp))

    return

if __name__ == '__main__':
    tutorial()