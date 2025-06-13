import numpy as np
from numba import njit

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append

from src.util.constants import G, msun, rsun, mearth, rearth

@njit
def s_dq(m1,m2,r1,wa,olap,rp):
    mt = m1+m2
    W22 = np.sqrt(6*np.pi/5)/2
    kp = wa*np.sqrt(rp**3/G/mt)
    z = np.sqrt(2.)*kp
    
    return 1j*kp*W22*olap*(m2/m1)*(r1/rp)**3 *2**1.5 \
        * 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))

@njit
def _rPL(mPL):
    x = np.log10(mPL/mearth)
    if mPL/mearth >=20.:
        y = -0.01043*x**3 - 0.002476*x**2 + 0.3036*x + 0.4603
    else:
        y = -0.02696*x**2 + 0.2827*x + 0.04447
    return 10**y*rearth

@njit
def _wf(mPL):
    return 1.3*np.sqrt(G*mPL/_rPL(mPL)**3) # Fuller 2014

@njit
def s_map_PLST_MT(m10,m20,wa,olap,q0,a0,rp,sigma,ga,mu1,ntot, q_break = 1.e-3, m1_min = 1.e-2, flag_vary_rp = False, flag_print_msg = False):
    qn = np.zeros(ntot, dtype=np.complex128)
    Pn = np.zeros(ntot)
    an = np.zeros(ntot)
    en = np.zeros(ntot)
    m1n = np.zeros(ntot)
    m2n = np.zeros(ntot)

    wan = np.zeros(ntot)
    
    mt = m10+m20
    r1 = _rPL(m10)
    Pn[0] = 2*np.pi*np.sqrt(a0**3/G/mt)
    qn[0] = q0*np.exp(-1j*wa*Pn[0]/2)
    an[0] = a0
    en[0] = 1-rp/a0

    m1n[0] = m10
    m2n[0] = m20

    rpn = rp
    wan[0] = _wf(m10)
    dq = s_dq(m1n[0],m2n[0],r1,wan[0],olap,rpn)

    for i in range(ntot-1):
        
        mt = m1n[i]+m2n[i]
        r1 = _rPL(m1n[i])
        if flag_vary_rp:
            rpn = an[i]*(1-en[i])
            dq = s_dq(m1n[i],m2n[i],r1,wan[i],olap,rpn)

        da_DT = -2*an[i]*(m1n[i]/m2n[i])*(an[i]/r1)*(np.abs(qn[i]+dq)**2-np.abs(qn[i])**2)
        de_DT = da_DT/an[i]*(1-en[i]**2)/2/en[i]

        da_MT, de_MT = 0., 0.
        dm1, dm2 = 0., 0.
        if np.abs(qn[i])**2 >= 0.1:
            q = m1n[i]/m2n[i]
            mun = m2n[i]/m1n[i]*mu1
            da_MT = 2*(-sigma)*(1+en[i])/(1-en[i])*((ga*q-1)+(1-ga)*(mun+0.5)*q/(1+q))*an[i]
            # da_MT = 2*(-sigma)*(1+en[i])/(1-en[i])*(-ga)*an[i]
            de_MT = (1-en[i])*da_MT/an[i]
            dm1 = -sigma*m1n[i]
            dm2 = -ga*dm1

        an[i+1] = an[i] + da_DT + da_MT
        en[i+1] = en[i] + de_DT + de_MT

        Pn[i+1] = 2*np.pi*np.sqrt(an[i+1]**3/G/mt)

        m1n[i+1] = m1n[i] + dm1
        m2n[i+1] = m2n[i] + dm2

        wan[i+1] = _wf(m1n[i+1])
        qn[i+1] = (qn[i]+dq)*np.exp(-1j*wan[i]*Pn[i+1])

        if np.abs(qn[i])**2 >= 0.1:
            qn[i+1] *= q_break

        if en[i+1]>1 or an[i+1]< 0.0 or m1n[i+1]<m1_min*m10:
            qn = qn[:i+2]
            an = an[:i+2]
            en = en[:i+2]
            m1n = m1n[:i+2]
            m2n = m2n[:i+2]
            Pn = Pn[:i+2]
            wan = wan[:i+2]
            if flag_print_msg:
                if en[i+1]>1:
                    print('e > 1')
                elif an[i+1]< 0.0:
                    print('a < 0')
                else:
                    print('m1 < threshold')
            break

    qn *= np.exp(-1j*wan*Pn/2)
    return qn, an, en, m1n, m2n

@njit
def s_map_PLST_MT_expo(m10,m20,wa,olap,q0,a0,rp,sigma,ga,mu1,ntot, alpha = 0.5 , q_break = 1.e-3, m1_min = 1.e-2, flag_vary_rp = False, flag_print_msg = False):
    qn = np.zeros(ntot, dtype=np.complex128)
    Pn = np.zeros(ntot)
    an = np.zeros(ntot)
    en = np.zeros(ntot)
    m1n = np.zeros(ntot)
    m2n = np.zeros(ntot)

    wan = np.zeros(ntot)
    
    mt = m10+m20
    r1 = _rPL(m10)
    Pn[0] = 2*np.pi*np.sqrt(a0**3/G/mt)
    qn[0] = q0*np.exp(-1j*wa*Pn[0]/2)
    an[0] = a0
    en[0] = 1-rp/a0

    m1n[0] = m10
    m2n[0] = m20

    rpn = rp
    wan[0] = _wf(m10)
    dq = s_dq(m1n[0],m2n[0],r1,wan[0],olap,rpn)

    for i in range(ntot-1):
        
        mt = m1n[i]+m2n[i]
        r1 = _rPL(m1n[i])
        if flag_vary_rp:
            rpn = an[i]*(1-en[i])
            dq = s_dq(m1n[i],m2n[i],r1,wan[i],olap,rpn)

        da_DT = -2*an[i]*(m1n[i]/m2n[i])*(an[i]/r1)*(np.abs(qn[i]+dq)**2-np.abs(qn[i])**2)
        de_DT = da_DT/an[i]*(1-en[i]**2)/2/en[i]

        da_MT, de_MT = 0., 0.
        dm1, dm2 = 0., 0.
        if np.abs(qn[i])**2 >= 0.1:
            rt = r1*(m2n[i]/m1n[i])**(1./3)
            q = m1n[i]/m2n[i]
            mun = m2n[i]/m1n[i]*mu1
            ga_expo = min(ga*np.exp(alpha*(2.-rpn/rt)),ga)
            de_MT = 2*(-sigma)*(1+en[i])*((ga_expo*q-1)+(1-ga_expo)*(mun+0.5)*q/(1+q))
            da_MT = de_MT/(1-(en[i]+de_MT))*an[i]           ### Important: In order to make rp the same before and after kick, we need to use the i+1-orbit for e in the denominator
            dm1 = -sigma*m1n[i]
            dm2 = -ga_expo*dm1

        an[i+1] = an[i] + da_DT + da_MT
        en[i+1] = en[i] + de_DT + de_MT

        Pn[i+1] = 2*np.pi*np.sqrt(an[i+1]**3/G/mt)

        m1n[i+1] = m1n[i] + dm1
        m2n[i+1] = m2n[i] + dm2

        wan[i+1] = _wf(m1n[i+1])
        qn[i+1] = (qn[i]+dq)*np.exp(-1j*wan[i]*Pn[i+1])

        if np.abs(qn[i])**2 >= 0.1:
            qn[i+1] = q_break*np.exp(1j*np.angle(qn[i+1]))

        if en[i+1]>1 or an[i+1]*(1-en[i+1])< 0.0 or m1n[i+1]<m1_min*m10 or en[i+1]<0.0:
            qn = qn[:i+2]
            an = an[:i+2]
            en = en[:i+2]
            m1n = m1n[:i+2]
            m2n = m2n[:i+2]
            Pn = Pn[:i+2]
            wan = wan[:i+2]
            if flag_print_msg:
                if en[i+1]>1:
                    print('e > 1')
                elif an[i+1]*(1-en[i+1])< 0.0:
                    print('rp < 0')
                elif m1n[i+1]<m1_min*m10:
                    print('m1 < threshold')
                else:
                    print('e < 0')
            break

    qn *= np.exp(-1j*wan*Pn/2)
    return qn, an, en, m1n, m2n

def tutorial():
    from src.util.constants import mearth, rearth, msun, rsun, AU
    import matplotlib.pyplot as plt
    
    m1 = 30.*mearth 
    r1 = _rPL(m1)
    m2 = 1.*msun
    rt = r1*(m2/m1)**(1./3)
    wa, olap = _wf(m1), 0.15
    a0 = 10*AU
    rp = 2.5*rt
    ntot = 500
    print('e = %1.3e'%(1-rp/a0))
    qn_mt, an_mt, en_mt, m1n_mt, m2n_mt = s_map_PLST_MT_expo(m1,m2,wa,olap\
                        ,q0=0.,a0 = a0, rp = rp,sigma=1.e-5,ga=1.,mu1=1.,ntot = ntot\
                        , alpha = 1., q_break= 1.e-3, m1_min = 0.1, flag_vary_rp = True)
    nlist = [x for x in range(len(qn_mt))]
    E1n_mt = G*(m1+m2)*m1/m2/r1*np.abs(qn_mt)**2
    
    plt.figure(figsize=(8.6,6.4), dpi= 100)
    plt.plot(nlist, np.abs(E1n_mt/(G*(m1+m2)/2/a0)), 'b+', label=r'$E_\text{mode, it, MT}$')
    plt.title(r'Mode energy $a_0$ = %1.2e'%(a0)+r'cm , $r_p = $ %1.2e'%rp+r'cm')
    plt.xlabel(r'$N_\text{orb}$',fontsize=15)
    plt.ylabel(r'$E/|E_\text{orb}|$',fontsize=15)
    # plt.yscale('log')
    plt.legend()
    plt.show()
    plt.close()

    return

if __name__ == '__main__':
    tutorial()