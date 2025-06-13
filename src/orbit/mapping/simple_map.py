import numpy as np
from numba import njit

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append

from src.util.constants import G

@njit
def s_dq(m1,m2,r1,wa,olap,rp):
    mt = m1+m2
    W22 = np.sqrt(6*np.pi/5)/2
    kp = wa*np.sqrt(rp**3/G/mt)
    z = np.sqrt(2.)*kp
    
    return 1j*kp*W22*olap*(m2/m1)*(r1/rp)**3 *2**1.5 \
        * 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))

@njit
def s_map(m1,m2,r1,wa,olap,q0,a0,rp,ntot, flag_vary_rp = False):
    qn = np.zeros(ntot, dtype=np.complex128)
    Pn = np.zeros(ntot)
    En = np.zeros(ntot)
    
    mt = m1+m2
    Pn[0] = 2*np.pi*np.sqrt(a0**3/G/mt)
    qn[0] = q0*np.exp(-1j*wa*Pn[0]/2)
    En[0] = -G*mt/2/a0

    dq = s_dq(m1,m2,r1,wa,olap,rp)

    '''
    Remark: The period in qn[i+1] must be Pn[i+1]
    Using Pn[i] will effectively introduce a damping in the mapping
    '''
    for i in range(ntot-1):
        if flag_vary_rp:
            a = -G*mt/2/En[i]
            e0 = 1- rp/a0
            e = np.sqrt(1-a0/a*(1-e0**2))
            rpn = a*(1-e)
            dq = s_dq(m1,m2,r1,wa,olap,rpn)

        dE = G*mt/r1*(m1/m2)*(np.abs(qn[i]+dq)**2-np.abs(qn[i])**2)
        En[i+1] = En[i] - dE
        Pn[i+1] = Pn[0]*(En[0]/En[i+1])**1.5
        # Pn[i+1] = Pn[i]-1.5*Pn[i]*(dE/En[i])
        qn[i+1] = (qn[i]+dq)*np.exp(-1j*wa*Pn[i+1])

    qn *= np.exp(-1j*wa*Pn/2)
    return qn, Pn


def tutorial():
    from src.util.constants import msun
    import matplotlib.pyplot as plt
    
    m1, r1 = 0.5*msun, 1.e10
    m2 = 1.e3*msun
    wa, olap = 0.3, 0.15
    a0 = 3.e12
    rp = 6.e10
    ntot = 500
    qn_s, P_s = s_map(m1,m2,r1,wa,olap\
                    ,q0=0., a0 = a0, rp = rp, ntot = ntot, flag_vary_rp= False)

    nlist = [x for x in range(ntot)]
    E1n_s = G*(m1+m2)*m1/m2/r1*np.abs(qn_s)**2
    
    plt.figure(figsize=(8.6,6.4), dpi= 100)
    plt.plot(nlist, np.abs(E1n_s/(G*(m1+m2)/2/a0)), 'rx', label=r'$E_\text{mode, it, simple}$')
    plt.title(r'Mode energy $a_0$ = %1.2e'%(a0)+r'cm , $r_p = $ %1.2e'%rp+r'cm')
    plt.xlabel(r'$N_\text{orb}$',fontsize=15)
    plt.ylabel(r'$E/|E_\text{orb}|$',fontsize=15)
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.close()

    return

if __name__ == '__main__':
    tutorial()