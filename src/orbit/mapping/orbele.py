import numpy as np
from numba import njit

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G, c
# from src.util.constants_Hang import G, c
from src.util.specialcoef import Wlm
from src.orbit.mapping.onekick_Lai97 import Ilm_steepest, Ilm_asymp_new
from src.orbit.mapping.Hansen import Hansen_direct
from src.orbit.osculating.orb_avg_formula import dP_1pn as dP_1pn_avg, dz_1pn as dz_1pn_avg
# from src.orbit.mapping.orbele_test_P1pn_corr import dP_1pn_corr_factor_num


def create_mode(w,olap,ga,ell,m):
    """  Create lists of modes in the form of multiple arrays """
    nn,nl,nm = len(w), len(ell), len(m)
    if len(olap) != nn:
        exit("Length of w differs from olap")

    mo_w, mo_olap, mo_ga, mo_ell, mo_m, mo_Wlm \
        = np.zeros(nn*nl*nm),np.zeros(nn*nl*nm),np.zeros(nn*nl*nm) \
            ,np.zeros(nn*nl*nm),np.zeros(nn*nl*nm),np.zeros(nn*nl*nm)

    ind = 0
    for i1 in range(nn):
        for i2 in range(nl):
            for i3 in range(nm):
                mo_w[ind] = w[i1]
                mo_olap[ind] = olap[i1]
                mo_ga[ind] = ga[i1]
                mo_ell[ind] = ell[i2]
                mo_m[ind] = m[i3]
                mo_Wlm[ind] = Wlm(ell[i2],m[i3])
                ind += 1
    return mo_w, mo_olap, mo_ga, mo_ell, mo_m, mo_Wlm

def convert_rp_1PN(x0):
    coef = np.array([2.9798,12.6526])
    return 1+coef[0]*x0+coef[1]*x0**2
def convert_rp_x1_1PN(x1):
    # coef = np.array([2.9798,3.7734])
    coef = np.array([2.9798,0.68282453])
    return 1+coef[0]*x1+coef[1]*x1**2
@njit
def modephase_mod(m1,m2,w,m,a,e,ga \
                , flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
                , flag_anharm = False, keff = 0., qn = [0.]\
                , flag_spin = False, ws = 0.):
    mt, eta = m1+m2, (m1*m2)/(m1+m2)**2
    p0K = 2*np.pi*np.sqrt(a**3/G/mt)

    dP_1pn = 0.
    if flag_1PN:
        dP_1pn = dP_1pn_avg(m1,m2,a,e)
    dP_2_5pn = 0.
    if flag_2_5PN:
        """ 
            In the mapping method, we update the period (at each apocenter). 
            This is different from the time it takes to go through each complete orbit.
            See Eq. (A6) of Vick and Lai (2018)
        """
        '''
            The 2.5PN effect is accounted for in da. No need to update P here.
        '''
        # p = a*(1-e**2)
        # dP_2_5pn = -192./5*np.pi*eta/(1-e**2)*(G*mt/p/c**2)**2.5*(1.+73./24.*e**2+37./96*e**4)
        dP_2_5pn = 0.0
    z = 1.
    if flag_redshift:
        # Ek = -G*mt/2/a
        # z += Ek/c**2*(m2/mt)*(2+m2/mt)
        z += dz_1pn_avg(m1,m2,a)
    dw_anharm = 0.
    if flag_anharm:
        for qni in qn:
            dw_anharm += -keff*np.abs(qni)**2
    
    dw_spin = np.zeros(len(w))
    if flag_spin:
        dw_spin += m*ws

    # wP = w*p0K*(z+dP_1pn+dP_2_5pn+dw_anharm)
    wP = (w+dw_spin)*p0K*(z+dP_1pn+dP_2_5pn+dw_anharm)
    # if flag_1PN:
        # print(z-1,dP_1pn, dP_1pn*1.015,wP)
        # print(z-1+dP_1pn, dP_1pn*0.015,wP)
    return wP

@njit
def dampphase_mod(m1,m2,damp,m,a,e,ga \
                , flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
                , flag_anharm = False, keff = 0., qn = [0.]):
    mt, eta = m1+m2, (m1*m2)/(m1+m2)**2
    p0K = 2*np.pi*np.sqrt(a**3/G/mt)

    dP_1pn = 0.
    if flag_1PN:
        dP_1pn = dP_1pn_avg(m1,m2,a,e)
    dP_2_5pn = 0.
    if flag_2_5PN:
        dP_2_5pn = 0.0
    z = 1.
    if flag_redshift:
        z += dz_1pn_avg(m1,m2,a)
    dw_anharm = 0.
    if flag_anharm:
        for qni in qn:
            dw_anharm += -keff*np.abs(qni)**2

    dampP = damp*p0K*(z+dP_1pn+dP_2_5pn+dw_anharm)
    return dampP

def dq_fixed(par,w,olap,ell,m,Wlm_in,role,Klm_flag=0\
            ,flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0., qn = [0.]\
            , flag_spin = False, ws = 0.):
    """  Compute one kick amplitude """
    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    # k = w*np.sqrt(a0**3/G/(m1+m2))
    k = modephase_mod(m1,m2,w,m,a0,e0,ga0, flag_1PN, flag_2_5PN, flag_redshift\
                      , flag_anharm, keff, qn, flag_spin, ws)/2/np.pi

    if role == 'primary':
        ep = Wlm_in*olap*(m2/m1)*(r1/a0)**(ell+1)+0j
    else:
        ep = Wlm_in*olap*(m1/m2)*(r2/a0)**(ell+1)*np.exp(-1j*m*np.pi)


    dga_1PN = 0.
    if flag_1PN:
        dga_1PN = 6*np.pi*G*(m1+m2)/c**2/a0/(1-e0**2)
    # ep = ep*np.exp(-1j*m*(ga0+dga_1PN))    
    ep = ep*np.exp(-1j*m*(ga0+dga_1PN/2))    

    y = np.sqrt(2.)*k*(1-e0)**1.5
    # return 1j*k*ep*2**1.5/(1-e0)**(ell-0.5)*Ilm_asymp_new(ell,-m,y)
    # return np.array([1j*k[i]*ep[i]*2**ell[i]*Ilm_steepest(y[i],ell[i],-m[i],eps=1.) for i in range(len(k))])
    if Klm_flag == 0:
        return np.array([1j*k[i]*ep[i]*2**1.5/(1-e0)**(ell[i]-0.5)*Ilm_steepest(y[i],ell[i],-m[i],eps=1.) for i in range(len(k))])
    else:
        return np.array([1j*k[i]*ep[i]*2*np.pi*Hansen_direct(ell[i],m[i],k[i],e0) for i in range(len(k))])

def dq_cal(par,w,olap,ell,m,Wlm_in,role\
            , flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0., qn = [0.]\
            , flag_spin = False, ws = 0.):

    if not flag_redshift and flag_1PN:
        coef = np.array([2.9798,12.6526,-0.7155,-1.0365])
    elif flag_redshift and not flag_1PN:
        coef = np.array([0.,0.,-1.5064,-1.9784])
        # coef = np.array([2.9798,12.6526,-2.4474,0.2907])
    elif flag_redshift and flag_1PN:
        coef = np.array([2.9798,12.6526,-2.1245,-0.4022])
    else:
        coef = np.zeros(4)

    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    k = modephase_mod(m1,m2,w,m,a0,e0,ga0, flag_1PN, flag_2_5PN, flag_redshift \
                      , flag_anharm, keff, qn, flag_spin, ws)/2/np.pi

    rp = a0*(1-e0)
    if role == 'primary':
        ep = Wlm_in*olap*(m2/m1)*(r1/rp)**(ell+1)+0j
    else:
        ep = Wlm_in*olap*(m1/m2)*(r2/rp)**(ell+1)*np.exp(-1j*m*np.pi)

    dga_1PN = 0.
    if flag_1PN:
        dga_1PN = 6*np.pi*G*(m1+m2)/c**2/a0/(1-e0**2)
    # ep = ep*np.exp(-1j*m*(ga0+dga_1PN))
    ep = ep*np.exp(-1j*m*(ga0+dga_1PN/2))      

    kp = k*(1-e0)**1.5
    y = np.sqrt(2.)*kp
    x = G*(m1+m2)/c**2/a0/(1-e0)

    amp = (1+coef[0]*x+coef[1]*x**2)**(ell+1)
    ymod = y*(1+coef[2]*x+coef[3]*x**2)

    return np.array([1j*kp[i]*ep[i]*2**1.5*amp[i]*Ilm_asymp_new(ell[i],-m[i],ymod[i]) for i in range(len(k))])
    # return np.array([1j*kp[i]*ep[i]*2**1.5*amp[i]*(2*np.pi**0.5/3*ymod[i]**1.5*np.exp(-2*ymod[i]/3)) for i in range(len(k))])


    # if role == 'primary':
    #     ep = Wlm_in*olap*(m2/m1)*(r1/a0)**(ell+1)+0j
    # else:
    #     ep = Wlm_in*olap*(m1/m2)*(r2/a0)**(ell+1)*np.exp(-1j*m*np.pi)

    # dga_1PN = 0.
    # if flag_1PN:
    #     dga_1PN = 6*np.pi*G*(m1+m2)/c**2/a0/(1-e0**2)
    # # ep = ep*np.exp(-1j*m*(ga0+dga_1PN))
    # ep = ep*np.exp(-1j*m*(ga0+dga_1PN/2))      

    # kp = k*(1-e0)**1.5
    # y = np.sqrt(2)*kp
    # x = G*(m1+m2)/c**2/a0/(1-e0)

    # amp = (1+coef[0]*x+coef[1]*x**2)**(ell+1)
    # ymod = y*(1+coef[2]*x+coef[3]*x**2)

    # return np.array([1j*k[i]*ep[i]*2**1.5/(1-e0)**(ell[i]-0.5)*amp[i]*Ilm_asymp_new(ell[i],-m[i],ymod[i]) for i in range(len(k))])

def dq_res(par,w,olap,ell,m,dW,phase):
    '''Resonance one-kick amplitude due to orbital period change'''
    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par
    W = np.sqrt(G*(m1+m2)/a0**3)
    beta = np.array([1j*w[i]*(m2/m1)*Wlm(ell[i],m[i])*olap[i]*(r1/a0)**(ell[i]+1) for i in range(len(w))])
    k = np.floor(w/W)
    Xlm = np.array([Hansen_direct(ell[i],m[i],k[i],e0) for i in range(len(k))])
    return np.array([beta[i]*np.exp(-1j*phase)*Xlm[i]*np.sqrt(2*np.pi/k[i]/dW)*np.exp(1j*np.pi/4) for i in range(len(w))])

@njit
def dq_eq(par,w,olap,ell,m,Wlm_in,role):
    """  Equilibrium tide change; From interaction energy """
    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    k = w*np.sqrt(a0**3/G/(m1+m2))
    if role == 'primary':
        ep = Wlm_in*olap*(m2/m1)*(r1/a0)**(ell+1)+0j
    else:
        ep = Wlm_in*olap*(m1/m2)*(r2/a0)**(ell+1)*np.exp(-1j*m*np.pi)

    return 2j *(-1.)**m *ep/(1.+e0)**(ell+1)*np.sin(np.pi*k)

@njit
def dorb(m1,r1,m2,r2\
        , a,e\
        , mo1_m, mo1_k, mo2_m, mo2_k\
        , dq1b: list[complex], q1: list[complex], dq2b: list[complex], q2: list[complex]\
        , dq1_eq: list[complex] = None, dq2_eq: list[complex] = None\
        ,flag_tide_reaction = True, flag_1PN = False, flag_2_5PN = False, flag_redshift = False):
    mtot = m1+m2
    mred = m1*m2/mtot

    if dq1_eq is None:
        dq1_eq = 0.*dq1b
    if dq2_eq is None:
        dq2_eq = 0.*dq2b

    if (flag_tide_reaction):
        z1, z2 = 1., 1.
        if flag_redshift:
            z1 += dz_1pn_avg(m1,m2,a)
            z2 += dz_1pn_avg(m2,m1,a)
        # da_DT = -2*a*(m1/m2)*(a/r1)*np.sum(np.real((dq1b-dq1_eq)*np.conj(dq1b+2*q1)))*z1 \
        #     -2*a*(m2/m1)*(a/r2)*np.sum(np.real((dq2b-dq2_eq)*np.conj(dq2b+2*q2)))*z2
        da_DT = -2*a*(m1/m2)*(a/r1)*np.sum(np.real(dq1b*np.conj(dq1b+2*q1)))*z1 \
            -2*a*(m2/m1)*(a/r2)*np.sum(np.real(dq2b*np.conj(dq2b+2*q2)))*z2
        # de_DT = -2*(m1/m2)*(a/r1)*(1-e**2)/2/e \
        #     *np.sum(np.real(((1-mo1_m/mo1_k/np.sqrt(1-e**2))*dq1b-dq1_eq)*np.conj(dq1b+2*q1)))*z1 \
        #     -2*(m2/m1)*(a/r2)*(1-e**2)/2/e \
        #     *np.sum(np.real(((1-mo2_m/mo2_k/np.sqrt(1-e**2))*dq2b-dq2_eq)*np.conj(dq2b+2*q2)))*z2
        de_DT = da_DT/a*(1-e**2)/2/e
        dga_DT = 0.
    else:
        da_DT = 0.
        de_DT = 0.
        dga_DT = 0.
    if (flag_1PN):
        dga_1PN = 6*np.pi*G*mtot/c**2/a/(1-e**2)
    else:
        dga_1PN = 0.

    if (flag_2_5PN):
        P0k = 2*np.pi*np.sqrt(a**3/G/(m1+m2))
        da_2_5PN = -64./5*(G**3/c**5)*(mtot**2*mred)/a**3*(1+73./24.*e**2+37./96*e**4)/(1-e**2)**3.5*P0k
        de_2_5PN = -32./5*(G**3/c**5)*(mtot**2*mred)/a**4*(19./6*e+121./96*e**3)/(1-e**2)**2.5*P0k
    else:
        da_2_5PN = 0.
        de_2_5PN = 0.

    da = da_DT + da_2_5PN
    de = de_DT + de_2_5PN
    dga = dga_DT + dga_1PN

    return da, de, dga

# @njit
# def dorb_DT(m1,r1,m2,r2, a,e\
#             , dq1: list[complex], q1: list[complex]\
#             , dq2: list[complex], q2: list[complex]\
#             , dq1_eq: list[complex] = None, dq2_eq: list[complex] = None):
#     if dq1_eq is None:
#         dq1_eq = 0.*dq1
#     if dq2_eq is None:
#         dq2_eq = 0.*dq2
#     # da = -2*a*(m1/m2)*(a/r1)*np.sum(np.real((dq1-dq1_eq)*np.conj(dq1+2*q1))) \
#     #     -2*a*(m2/m1)*(a/r2)*np.sum(np.real((dq2-dq2_eq)*np.conj(dq2+2*q2)))
#     da = -2*a*(m1/m2)*(a/r1)*np.sum(np.real(dq1*np.conj(dq1+2*q1))) \
#         -2*a*(m2/m1)*(a/r2)*np.sum(np.real(dq2*np.conj(dq2+2*q2)))
#     de = da/a*(1-e**2)/2/e
#     return da, de

@njit
def one_phase_change(par: list[float]\
            , mo1_w: list[float], mo1_m: list[int], q10: list[complex], dq1: list[complex]\
            , mo2_w: list[float], mo2_m: list[int], q20: list[complex], dq2: list[complex]\
            , dq1_eq: list[complex] = None, dq2_eq: list[complex] = None\
            , flag_tide_reaction = True, flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0.\
            , flag_spin = False, ws10 = 0., ws20 = 0., dws1 = 0., dws2 = 0.):
    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    mo1_phase_0 = modephase_mod(m1,m2,mo1_w,mo1_m,a0,e0,ga0,flag_1PN,flag_2_5PN,flag_redshift\
                            , flag_anharm, keff, qn = q10, flag_spin=flag_spin, ws = ws10)
    mo2_phase_0 = modephase_mod(m2,m1,mo2_w,mo2_m,a0,e0,ga0,flag_1PN,flag_2_5PN,flag_redshift\
                              , flag_anharm, keff, qn = q20, flag_spin=flag_spin, ws = ws20)

    da, de, dga = dorb(m1,r1,m2,r2, a0, e0, mo1_m, mo1_phase_0, mo2_m, mo2_phase_0,dq1,q10,dq2,q20 \
                        ,dq1_eq,dq2_eq,flag_tide_reaction,flag_1PN,flag_2_5PN,flag_redshift)

    a1, e1, ga1 = a0+da, e0+de, ga0+dga
    ws11, ws21 = ws10 + dws1, ws20 + dws2

    mo1_phase_1 = modephase_mod(m1,m2,mo1_w,mo1_m,a1,e1,ga1,flag_1PN,flag_2_5PN,flag_redshift\
                            , flag_anharm, keff, qn = q10, flag_spin=flag_spin, ws = ws11)
    mo2_phase_1 = modephase_mod(m2,m1,mo2_w,mo2_m,a1,e1,ga1,flag_1PN,flag_2_5PN,flag_redshift\
                              , flag_anharm, keff, qn = q20, flag_spin=flag_spin, ws = ws21)

    return (mo1_phase_1-mo1_phase_0), (mo2_phase_1-mo2_phase_0)

@njit
def iterate(par: list[float]\
            , mo1_w: list[float], mo1_ga: list[float], mo1_m: list[int], q10: list[complex], dq1: list[complex]\
            , mo2_w: list[float], mo2_ga: list[float], mo2_m: list[int], q20: list[complex], dq2: list[complex]\
            , ntot\
            , dq1_eq: list[complex] = None, dq2_eq: list[complex] = None\
            , flag_tide_reaction = True, flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0.\
            , flag_spin = False, ws10 = 0., ws20 = 0., dws1 = 0., dws2 = 0.):

    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    an, en, gan = np.zeros(ntot), np.zeros(ntot), np.zeros(ntot)
    q1n = np.array([[0.]*len(q10)]*ntot, dtype = np.complex128)
    q2n = np.array([[0.]*len(q20)]*ntot, dtype = np.complex128)
    ws1n, ws2n = np.zeros(ntot), np.zeros(ntot)

    an[0], en[0], gan[0] = a0, e0, ga0
    ws1n[0], ws2n[0] = ws10, ws20

    # Shift the mode amplitudes for convenience in mapping
    mo1_phase = modephase_mod(m1,m2,mo1_w,mo1_m,a0,e0,ga0,flag_1PN,flag_2_5PN,flag_redshift\
                              , flag_anharm, keff, qn = q10, flag_spin=flag_spin, ws = ws10)
    mo2_phase = modephase_mod(m2,m1,mo2_w,mo2_m,a0,e0,ga0,flag_1PN,flag_2_5PN,flag_redshift\
                              , flag_anharm, keff, qn = q20, flag_spin=flag_spin, ws = ws20) 
    
    """Shift for half period for the mapping method. Eq. (A10) of Vick and Lai (2018)"""   
    q1n[0,:], q2n[0,:] = q10*np.exp(-1j*mo1_phase/2), q20*np.exp(-1j*mo2_phase/2)

    # mo1_old = mo1_phase
    # mo2_old = mo2_phase

    for i in range(ntot-1):
        dq1b, dq2b= dq1*np.exp(-1j*mo1_m[:]*gan[i]), dq2*np.exp(-1j*mo2_m[:]*gan[i])
        da, de, dga = dorb(m1,r1,m2,r2, an[i],en[i], mo1_m, mo1_phase, mo2_m, mo2_phase \
                           ,dq1b[:],q1n[i,:],dq2b[:],q2n[i,:] \
                           ,dq1_eq,dq2_eq,flag_tide_reaction,flag_1PN,flag_2_5PN,flag_redshift)

        an[i+1], en[i+1], gan[i+1] = an[i]+da, en[i]+de, gan[i]+dga
        ws1n[i+1], ws2n[i+1] = ws1n[i]+dws1, ws2n[i]+dws2

        # q1n[i+1,:] = q_n(q1n[i,:],dq1[:],m1,m2,mo1_w[:],mo1_m[:],an[i],en[i],gan[i],dga\
        #                  ,flag_1PN,flag_2_5PN,flag_redshift,flag_anharm, keff)
        # q2n[i+1,:] = q_n(q2n[i,:],dq2[:],m2,m1,mo2_w[:],mo2_m[:],an[i],en[i],gan[i],dga\
        #                  ,flag_1PN,flag_2_5PN,flag_redshift, flag_anharm, keff)
        
        '''Remark: It is important to use [i+1] component of the orbital elements.
            This follows from the solution of qb[i+1] = (qb[i]+dq)*exp(-i w P[i+1]) '''
        mo1_phase = modephase_mod(m1,m2,mo1_w[:],mo1_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                , flag_anharm, keff, qn = q1n[i,:]\
                                , flag_spin=flag_spin, ws = ws1n[i+1])
        mo1_damp = dampphase_mod(m1,m2,mo1_ga[:],mo1_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                ,flag_anharm, keff, qn = q1n[i,:])
        # q1n[i+1,:] =  (q1n[i,:]+dq1[:]*np.exp(-1j*mo1_m[:]*gan[i])*np.exp(-1j*(-mo1_phase+mo1_old)/4))*np.exp(-1j*mo1_phase)
        q1n[i+1,:] =  (q1n[i,:]+dq1[:]*np.exp(-1j*mo1_m[:]*gan[i]))*np.exp(-1j*mo1_phase)*np.exp(-mo1_damp)
        mo2_phase = modephase_mod(m2,m1,mo2_w[:],mo2_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                , flag_anharm, keff, qn = q2n[i,:]\
                                , flag_spin=flag_spin, ws = ws2n[i+1])
        mo2_damp = dampphase_mod(m2,m1,mo2_ga[:],mo2_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                ,flag_anharm, keff, qn = q2n[i,:])
        # q2n[i+1,:] =  (q2n[i,:]+dq2[:]*np.exp(-1j*mo2_m[:]*gan[i])*np.exp(-1j*(-mo2_phase+mo2_old)/4))*np.exp(-1j*mo2_phase)
        q2n[i+1,:] =  (q2n[i,:]+dq2[:]*np.exp(-1j*mo2_m[:]*gan[i]))*np.exp(-1j*mo2_phase)*np.exp(-mo2_damp)
        # mo1_old = mo1_phase
        # mo2_old = mo2_phase

    # Shift the mode amplitudes back
    for i in range(ntot):
        # p0K = 2*np.pi*np.sqrt(an[i]**3/G/(m1+m2))
        # q1n[i,:]=q1n[i,:]*np.exp(1j*mo1_w*p0K/2)
        # q2n[i,:]=q2n[i,:]*np.exp(1j*mo2_w*p0K/2)
        mo1_phase = modephase_mod(m1,m2,mo1_w,mo1_m,an[i],en[i],gan[i] \
                                ,flag_1PN,flag_2_5PN,flag_redshift, flag_anharm, keff, qn = q1n[i,:]\
                                ,flag_spin=flag_spin, ws = ws1n[i])
        mo2_phase = modephase_mod(m2,m1,mo2_w,mo2_m,an[i],en[i],gan[i] \
                                  ,flag_1PN,flag_2_5PN,flag_redshift, flag_anharm, keff, qn = q2n[i,:]\
                                ,flag_spin=flag_spin, ws = ws2n[i])
        """Shift back for the actual mode amplitude"""
        q1n[i,:]=q1n[i,:]*np.exp(1j*mo1_phase/2)
        q2n[i,:]=q2n[i,:]*np.exp(1j*mo2_phase/2)
    return an, en, gan, q1n, q2n

#====================================================================================
'''Iterative map that updates dq'''
@njit
def _Ilm_asymp_new(ell,m,z):
    """ Rederived by me using the steepest descent method """
    if ell == 2:
        if m == -2:
            Ilm = 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))
        elif m == 0:
            Ilm = np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z)+101./144/z)
        elif m == 2:
            Ilm = np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-17./12/z)
        else:
            Ilm = 0.
    else:
        Ilm = 0.
    return Ilm

@njit
def dq_fixed_2(par,w,olap,ell,m,Wlm_in,role\
            ,flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0., qn = [0.]\
            , flag_spin = False, ws = 0.):
    """  Compute one kick amplitude """
    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    k = modephase_mod(m1,m2,w,m,a0,e0,ga0, flag_1PN, flag_2_5PN, flag_redshift\
                      , flag_anharm, keff, qn, flag_spin, ws)/2/np.pi

    if role == 'primary':
        ep = Wlm_in*olap*(m2/m1)*(r1/a0)**(ell+1)+0j
    else:
        ep = Wlm_in*olap*(m1/m2)*(r2/a0)**(ell+1)*np.exp(-1j*m*np.pi)


    dga_1PN = 0.
    if flag_1PN:
        dga_1PN = 6*np.pi*G*(m1+m2)/c**2/a0/(1-e0**2)
    ep = ep*np.exp(-1j*m*(ga0+dga_1PN/2))    

    y = np.sqrt(2.)*k*(1-e0)**1.5
    return np.array([1j*k[i]*ep[i]*2**1.5/(1-e0)**(ell[i]-0.5)*_Ilm_asymp_new(ell[i],-m[i],y[i]) for i in range(len(k))])

@njit
def dq_cal_2(par,w,olap,ell,m,Wlm_in,role\
            , flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0., qn = [0.]\
            , flag_spin = False, ws = 0.):

    if not flag_redshift and flag_1PN:
        coef = np.array([2.9798,12.6526,-0.7155,-1.0365])
    elif flag_redshift and not flag_1PN:
        coef = np.array([0.,0.,-1.5064,-1.9784])
        # coef = np.array([2.9798,12.6526,-2.4474,0.2907])
    elif flag_redshift and flag_1PN:
        coef = np.array([2.9798,12.6526,-2.1245,-0.4022])
    else:
        coef = np.zeros(4)

    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    k = modephase_mod(m1,m2,w,m,a0,e0,ga0, flag_1PN, flag_2_5PN, flag_redshift \
                      , flag_anharm, keff, qn, flag_spin, ws)/2/np.pi

    rp = a0*(1-e0)
    if role == 'primary':
        ep = Wlm_in*olap*(m2/m1)*(r1/rp)**(ell+1)+0j
    else:
        ep = Wlm_in*olap*(m1/m2)*(r2/rp)**(ell+1)*np.exp(-1j*m*np.pi)

    dga_1PN = 0.
    if flag_1PN:
        dga_1PN = 6*np.pi*G*(m1+m2)/c**2/a0/(1-e0**2)
    ep = ep*np.exp(-1j*m*(ga0+dga_1PN/2))      

    kp = k*(1-e0)**1.5
    y = np.sqrt(2.)*kp
    x = G*(m1+m2)/c**2/a0/(1-e0)

    amp = (1+coef[0]*x+coef[1]*x**2)**(ell+1)
    ymod = y*(1+coef[2]*x+coef[3]*x**2)

    return np.array([1j*kp[i]*ep[i]*2**1.5*amp[i]*_Ilm_asymp_new(ell[i],-m[i],ymod[i]) for i in range(len(k))])

@njit
def update_dq(par_input\
            ,mo1_w, mo1_olap, mo1_ell, mo1_m, mo1_Wlm\
            ,mo2_w, mo2_olap, mo2_ell, mo2_m, mo2_Wlm\
            ,flag_dq_cal\
            ,flag_1PN, flag_2_5PN, flag_redshift, flag_anharm\
            ,keff, q1n, q2n\
            ,ws1, ws2):
    if flag_dq_cal:
        dq1 = dq_cal_2(par_input, mo1_w, mo1_olap, mo1_ell, mo1_m, mo1_Wlm\
                    ,role='primary' \
                    ,flag_1PN= flag_1PN\
                    ,flag_2_5PN= flag_2_5PN\
                    ,flag_redshift= flag_redshift\
                    ,flag_anharm = flag_anharm, keff = keff, qn = q1n\
                    ,ws=ws1)
        dq2 = dq_cal_2(par_input, mo2_w, mo2_olap, mo2_ell, mo2_m, mo2_Wlm\
                    ,role='secondary' \
                    ,flag_1PN= flag_1PN\
                    ,flag_2_5PN= flag_2_5PN\
                    ,flag_redshift= flag_redshift\
                    ,flag_anharm = flag_anharm, keff = keff, qn = q2n\
                    ,ws=ws2)
    else:
        dq1 = dq_fixed_2(par_input, mo1_w, mo1_olap, mo1_ell, mo1_m, mo1_Wlm\
                    ,role='primary' \
                    ,flag_1PN= flag_1PN\
                    ,flag_2_5PN= flag_2_5PN\
                    ,flag_redshift= flag_redshift\
                    ,flag_anharm = flag_anharm, keff = keff, qn = q1n\
                    ,ws=ws1)
        dq2 = dq_fixed_2(par_input, mo2_w, mo2_olap, mo2_ell, mo2_m, mo2_Wlm\
                    ,role='secondary' \
                    ,flag_1PN= flag_1PN\
                    ,flag_2_5PN= flag_2_5PN\
                    ,flag_redshift= flag_redshift\
                    ,flag_anharm = flag_anharm, keff = keff, qn = q2n\
                    ,ws=ws2)
    return dq1, dq2

@njit
def iterate2(par: list[float]\
            , mo1_w: list[float], mo1_ga: list[float], mo1_olap: list[float], mo1_ell: list[float], mo1_m: list[int], mo1_Wlm: list[int], q10: list[complex]\
            , mo2_w: list[float], mo2_ga: list[float], mo2_olap: list[float], mo2_ell: list[float], mo2_m: list[int], mo2_Wlm: list[int], q20: list[complex]\
            , ntot\
            , dq1_eq: list[complex] = None, dq2_eq: list[complex] = None\
            , flag_dq_cal = False\
            , flag_tide_reaction = True, flag_1PN = False, flag_2_5PN = False, flag_redshift = False\
            , flag_anharm = False, keff = 0.\
            , flag_spin = False, ws10 = 0., ws20 = 0., dws1 = 0., dws2 = 0.):

    m1,r1,m2,r2 \
    ,a0, e0, ga0 \
    = par

    an, en, gan = np.zeros(ntot), np.zeros(ntot), np.zeros(ntot)
    q1n = np.array([[0.]*len(q10)]*ntot, dtype = np.complex128)
    q2n = np.array([[0.]*len(q20)]*ntot, dtype = np.complex128)
    ws1n, ws2n = np.zeros(ntot), np.zeros(ntot)

    an[0], en[0], gan[0] = a0, e0, ga0
    ws1n[0], ws2n[0] = ws10, ws20

    # Shift the mode amplitudes for convenience in mapping
    mo1_phase = modephase_mod(m1,m2,mo1_w,mo1_m,a0,e0,ga0,flag_1PN,flag_2_5PN,flag_redshift\
                              , flag_anharm, keff, qn = q10, flag_spin=flag_spin, ws = ws10)
    mo2_phase = modephase_mod(m2,m1,mo2_w,mo2_m,a0,e0,ga0,flag_1PN,flag_2_5PN,flag_redshift\
                              , flag_anharm, keff, qn = q20, flag_spin=flag_spin, ws = ws20) 
    """Shift for half period for the mapping method. Eq. (A10) of Vick and Lai (2018)"""   
    q1n[0,:], q2n[0,:] = q10*np.exp(-1j*mo1_phase/2), q20*np.exp(-1j*mo2_phase/2)

    """Find one kick amplitude"""
    dq1, dq2 = update_dq(par\
            ,mo1_w, mo1_olap, mo1_ell, mo1_m, mo1_Wlm\
            ,mo2_w, mo2_olap, mo2_ell, mo2_m, mo2_Wlm\
            ,flag_dq_cal\
            ,flag_1PN, flag_2_5PN, flag_redshift, flag_anharm\
            ,keff, q1n[0,:], q2n[0,:]\
            ,ws10,ws20)

    """Test feature: Equilibrium tide correction to da"""
    # dq1_eq = dq_eq(par,mo1_w,mo1_olap,mo1_ell,mo1_m,mo1_Wlm,role='primary')
    # dq2_eq = dq_eq(par,mo2_w,mo2_olap,mo2_ell,mo2_m,mo2_Wlm,role='secondary')

    for i in range(ntot-1):
        dq1b, dq2b= dq1*np.exp(-1j*mo1_m[:]*gan[i]), dq2*np.exp(-1j*mo2_m[:]*gan[i])
        da, de, dga = dorb(m1,r1,m2,r2, an[i],en[i], mo1_m, mo1_phase, mo2_m, mo2_phase\
                           ,dq1b[:],q1n[i,:],dq2b[:],q2n[i,:] \
                           ,dq1_eq,dq2_eq,flag_tide_reaction,flag_1PN,flag_2_5PN,flag_redshift)

        an[i+1], en[i+1], gan[i+1] = an[i]+da, en[i]+de, gan[i]+dga
        ws1n[i+1], ws2n[i+1] = ws1n[i]+dws1, ws2n[i]+dws2
      
        mo1_phase = modephase_mod(m1,m2,mo1_w[:],mo1_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                , flag_anharm, keff, qn = q1n[i,:]\
                                , flag_spin=flag_spin, ws = ws1n[i+1])
        mo1_damp = dampphase_mod(m1,m2,mo1_ga[:],mo1_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                ,flag_anharm, keff, qn = q1n[i,:])

        q1n[i+1,:] =  (q1n[i,:]+dq1[:]*np.exp(-1j*mo1_m[:]*gan[i]))*np.exp(-1j*mo1_phase)*np.exp(-mo1_damp)
        mo2_phase = modephase_mod(m2,m1,mo2_w[:],mo2_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                , flag_anharm, keff, qn = q2n[i,:]\
                                , flag_spin=flag_spin, ws = ws2n[i+1])
        mo2_damp = dampphase_mod(m2,m1,mo2_ga[:],mo2_m[:],an[i+1],en[i+1],gan[i+1]\
                                ,flag_1PN,flag_2_5PN,flag_redshift\
                                ,flag_anharm, keff, qn = q2n[i,:])
        
        q2n[i+1,:] =  (q2n[i,:]+dq2[:]*np.exp(-1j*mo2_m[:]*gan[i]))*np.exp(-1j*mo2_phase)*np.exp(-mo2_damp)

        par_n = np.array([m1,r1,m2,r2,an[i+1], en[i+1], gan[i+1]])

        dq1, dq2 = update_dq(par_n\
                            ,mo1_w, mo1_olap, mo1_ell, mo1_m, mo1_Wlm\
                            ,mo2_w, mo2_olap, mo2_ell, mo2_m, mo2_Wlm\
                            ,flag_dq_cal\
                            ,flag_1PN, flag_2_5PN, flag_redshift, flag_anharm\
                            ,keff, q1n[i+1,:], q2n[i+1,:]\
                            ,ws1n[i+1],ws2n[i+1])
        
        # dq1_eq = dq_eq(par_n,mo1_w,mo1_olap,mo1_ell,mo1_m,mo1_Wlm,role='primary')
        # dq2_eq = dq_eq(par_n,mo2_w,mo2_olap,mo2_ell,mo2_m,mo2_Wlm,role='secondary')

    # Shift the mode amplitudes back
    for i in range(ntot):
        mo1_phase = modephase_mod(m1,m2,mo1_w,mo1_m,an[i],en[i],gan[i] \
                                ,flag_1PN,flag_2_5PN,flag_redshift, flag_anharm, keff, qn = q1n[i,:]\
                                ,flag_spin=flag_spin, ws = ws1n[i])
        mo2_phase = modephase_mod(m2,m1,mo2_w,mo2_m,an[i],en[i],gan[i] \
                                  ,flag_1PN,flag_2_5PN,flag_redshift, flag_anharm, keff, qn = q2n[i,:]\
                                ,flag_spin=flag_spin, ws = ws2n[i])
        """Shift back for the actual mode amplitude"""
        q1n[i,:]=q1n[i,:]*np.exp(1j*mo1_phase/2)
        q2n[i,:]=q2n[i,:]*np.exp(1j*mo2_phase/2)
    return an, en, gan, q1n, q2n

#====================================================================================
'''Wrapper of the iterative map method'''
class itmap:
    def __init__(self):
        self.par = {}
        self.mo1 = {}
        self.q10_factor = 0.
        self.mo2 = {}
        self.q20_factor = 0.
        self.Klm_flag = 0
        self.flag_tide_reaction = True
        self.flag_1pn = False
        self.flag_2_5pn = False
        self.flag_redshift = False
        self.flag_dq_cal = False
        
        self.flag_anharm = False
        self.keff = 1.

        self.flag_spin = False
        self.ws10 = 0.
        self.dws1 = 0.
        self.ws20 = 0.
        self.dws2 = 0.

    def get_mode_1(self,w,olap,ell,m, q0 = None, ga = None):
        if ga == None:
            ga = np.zeros(len(w))
        
        w, olap, ga, ell, m, Wlm = create_mode(w,olap,ga,ell,m)

        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        self.mo1['w'],self.mo1['olap'],self.mo1['ga'],self.mo1['ell'],self.mo1['m'],self.mo1['Wlm'] \
        = w, olap, ga, ell, m, Wlm
        
        if q0 == None:
            self.mo1['q0'] = np.ones(len(w))*self.q10_factor
        else:
            self.mo1['q0'] = q0

        if self.flag_dq_cal:
            self.mo1['dq'] = dq_cal(par_vec, w, olap, ell, m, Wlm\
                            ,role='primary' \
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo1['q0'])
        else:
            self.mo1['dq'] = dq_fixed(par_vec, w, olap, ell, m, Wlm\
                            ,role='primary' \
                            ,Klm_flag= self.Klm_flag\
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo1['q0'])

    def get_mode_2(self,w,olap,ell,m, q0 = None, ga = None):
        if ga == None:
            ga = np.zeros(len(w))

        w, olap, ga, ell, m, Wlm = create_mode(w,olap,ga,ell,m)

        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        self.mo2['w'],self.mo2['olap'],self.mo2['ga'],self.mo2['ell'],self.mo2['m'],self.mo2['Wlm'] \
        = w, olap, ga, ell, m, Wlm

        if q0 == None:
            self.mo2['q0'] = np.ones(len(w))*self.q20_factor
        else:
            self.mo2['q0'] = q0

        if self.flag_dq_cal:
            self.mo2['dq'] = dq_cal(par_vec, w, olap, ell, m, Wlm\
                            ,role='secondary' \
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo2['q0'])
        else:
            self.mo2['dq'] = dq_fixed(par_vec, w, olap, ell, m, Wlm\
                            ,role='secondary' \
                            ,Klm_flag= self.Klm_flag\
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo2['q0'])

    def get_dq1(self,w,olap,ell,m, q0 = None):
        ga = np.zeros(len(w))
        w, olap, ga, ell, m, Wlm = create_mode(w,olap,ga,ell,m)

        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        if q0 == None:
            self.mo1['q0'] = np.ones(len(w))*self.q10_factor
        else:
            self.mo1['q0'] = q0

        if self.flag_dq_cal:
            dq = dq_cal(par_vec, w, olap, ell, m, Wlm\
                            ,role='primary' \
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo1['q0'])
        else:
            dq = dq_fixed(par_vec, w, olap, ell, m, Wlm\
                            ,role='primary' \
                            ,Klm_flag= self.Klm_flag\
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo1['q0'])
        return dq

    def get_dq2(self,w,olap,ell,m, q0 = None):
        ga = np.zeros(len(w))
        w, olap, ga, ell, m, Wlm = create_mode(w,olap,ga,ell,m)

        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])
        
        if q0 == None:
            self.mo2['q0'] = np.ones(len(w))*self.q20_factor
        else:
            self.mo2['q0'] = q0

        if self.flag_dq_cal:
            dq = dq_cal(par_vec, w, olap, ell, m, Wlm\
                            ,role='secondary' \
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo2['q0'])
        else:
            dq = dq_fixed(par_vec, w, olap, ell, m, Wlm\
                            ,role='secondary' \
                            ,Klm_flag= self.Klm_flag\
                            ,flag_1PN= self.flag_1pn\
                            ,flag_2_5PN= self.flag_2_5pn\
                            ,flag_redshift= self.flag_redshift\
                            , flag_anharm = self.flag_anharm, keff = self.keff, qn = self.mo2['q0'])
        return dq

    def get_dq_res(self,w,olap,ell,m,dW,phase):
        ga = np.zeros(len(w))
        w, olap, ga, ell, m, Wlm = create_mode(w,olap,ga,ell,m)
        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        return dq_res(par_vec,w,olap,ell,m,dW,phase)

    @staticmethod
    def convert_rp_1PN(mt,rp):
        return convert_rp_1PN(G*mt/c**2/rp)
    
    @staticmethod
    def convert_rp_x1_1PN(mt,rp1):
        return convert_rp_x1_1PN(G*mt/c**2/rp1)
    
    def one_dphase(self):
        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        mo1_w, mo1_m, q10, dq1 = self.mo1['w'], self.mo1['m'], self.mo1['q0'], self.mo1['dq']
        mo2_w, mo2_m, q20, dq2 = self.mo2['w'], self.mo2['m'], self.mo2['q0'], self.mo2['dq']

        self.dphase1, self.dphase2 = one_phase_change(par_vec, mo1_w, mo1_m, q10, dq1, mo2_w, mo2_m, q20, dq2\
                                            ,dq1_eq=None,dq2_eq=None \
                                            ,flag_tide_reaction = self.flag_tide_reaction \
                                            ,flag_1PN= self.flag_1pn\
                                            ,flag_2_5PN= self.flag_2_5pn\
                                            ,flag_redshift= self.flag_redshift\
                                            ,flag_anharm = self.flag_anharm, keff = self.keff\
                                            ,flag_spin = self.flag_spin, ws10 = self.ws10, ws20 = self.ws20\
                                            ,dws1 = self.dws1, dws2 = self.dws2)

    def map(self,ntot):

        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        mo1_w, mo1_ga, mo1_m, q10, dq1 = self.mo1['w'], self.mo1['ga'], self.mo1['m'], self.mo1['q0'], self.mo1['dq']
        mo2_w, mo2_ga, mo2_m, q20, dq2 = self.mo2['w'], self.mo2['ga'], self.mo2['m'], self.mo2['q0'], self.mo2['dq']

        an, en, gan, q1n, q2n \
        = iterate(par_vec, mo1_w, mo1_ga, mo1_m, q10, dq1\
                ,mo2_w, mo2_ga, mo2_m, q20, dq2, ntot\
                ,dq1_eq=None,dq2_eq=None \
                ,flag_tide_reaction = self.flag_tide_reaction \
                ,flag_1PN= self.flag_1pn\
                ,flag_2_5PN= self.flag_2_5pn\
                ,flag_redshift= self.flag_redshift\
                ,flag_anharm = self.flag_anharm, keff = self.keff\
                ,flag_spin = self.flag_spin, ws10 = self.ws10, ws20 = self.ws20\
                ,dws1 = self.dws1, dws2 = self.dws2)
        
        z1n, z2n = 1., 1.
        if self.flag_redshift:
            z1n += dz_1pn_avg(m1,m2,an)
            z2n += dz_1pn_avg(m2,m1,an)
        E1n = G*(m1+m2)*(m1/m2)/r1*np.sum(np.abs(q1n)**2,axis = 1)*z1n
        E2n = G*(m1+m2)*(m2/m1)/r2*np.sum(np.abs(q2n)**2,axis = 1)*z2n

        self.an, self.en, self.gan, self.q1n, self.q2n = an, en, gan, q1n, q2n
        self.E1n, self.E2n = E1n, E2n

    def map2(self,ntot):

        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        a0, e0, ga0 = self.par['a0'],self.par['e0'],self.par['ga0']
        par_vec = np.array([m1,r1,m2,r2,a0,e0,ga0])

        mo1_w, mo1_ga, mo1_olap, mo1_ell, mo1_m, mo1_Wlm, q10 \
            = self.mo1['w'], self.mo1['ga'], self.mo1['olap'], self.mo1['ell'], self.mo1['m'], self.mo1['Wlm'], self.mo1['q0']
        mo2_w, mo2_ga, mo2_olap, mo2_ell, mo2_m, mo2_Wlm, q20 \
            = self.mo2['w'], self.mo2['ga'], self.mo2['olap'], self.mo2['ell'], self.mo2['m'], self.mo2['Wlm'], self.mo2['q0']

        an, en, gan, q1n, q2n \
        = iterate2(par_vec, mo1_w, mo1_ga, mo1_olap, mo1_ell, mo1_m, mo1_Wlm, q10\
                ,mo2_w, mo2_ga, mo2_olap, mo2_ell, mo2_m, mo2_Wlm, q20\
                ,ntot\
                ,dq1_eq=None,dq2_eq=None \
                ,flag_dq_cal = self.flag_dq_cal\
                ,flag_tide_reaction = self.flag_tide_reaction \
                ,flag_1PN= self.flag_1pn\
                ,flag_2_5PN= self.flag_2_5pn\
                ,flag_redshift= self.flag_redshift\
                ,flag_anharm = self.flag_anharm, keff = self.keff\
                ,flag_spin = self.flag_spin, ws10 = self.ws10, ws20 = self.ws20\
                ,dws1 = self.dws1, dws2 = self.dws2)


        z1n, z2n = 1., 1.
        if self.flag_redshift:
            z1n += dz_1pn_avg(m1,m2,an)
            z2n += dz_1pn_avg(m2,m1,an)
        E1n = G*(m1+m2)*(m1/m2)/r1*np.sum(np.abs(q1n)**2,axis = 1)*z1n
        E2n = G*(m1+m2)*(m2/m1)/r2*np.sum(np.abs(q2n)**2,axis = 1)*z2n

        self.an, self.en, self.gan, self.q1n, self.q2n = an, en, gan, q1n, q2n
        self.E1n, self.E2n = E1n, E2n

#====================================================================================
'''Example codes'''
def tutorial():
    from src.util.constants import msun
    import matplotlib.pyplot as plt
    # import scienceplots
    
    ntot = 1500

    mo1_w,mo1_olap,mo1_ga,mo1_ell,mo1_m, mo1_Wlm = create_mode(w=[0.2],olap=[0.5],ga=[0.],ell=[2],m=[0])
    mo2_w,mo2_olap,mo2_ga,mo2_ell,mo2_m, mo2_Wlm = create_mode(w=[],olap=[],ga=[],ell=[],m=[])

    m1, r1 = 0.4*msun, 1.e9
    m2, r2 = 0.6*msun, 6.e8
    a0, e0, ga0 = 2.e9, 0.95, 0.
    par = np.array([m1,r1,m2,r2\
                    ,a0,e0,ga0])
    
    dq1 = dq_fixed(par,mo1_w,mo1_olap,mo1_ell,mo1_m,mo1_Wlm,'primary', flag_1PN= False, flag_2_5PN= False)
    q10 = 0.*dq1
    dq2 = np.array([])
    q20 = 0.*dq2

    dq1_eq = dq_eq(par,mo1_w,mo1_olap,mo1_ell,mo1_m,mo1_Wlm,'primary')

    # k21 = np.sum((mo1_Wlm*mo1_olap)**2)
    # k22 = np.sum((mo2_Wlm*mo2_olap)**2)
    # @njit
    # def _dga_eqtide(a,e,ga):
    #     return 30*np.pi*(m2/m1)*k21*(r1/a)**5*(1+1.5*e**2+0.125*e**4) \
    #         + 30*np.pi*(m1/m2)*k22*(r2/a)**5*(1+1.5*e**2+0.125*e**4)

    an, en, gan, q1n, q2n \
        = iterate(par, mo1_w, mo1_ga, mo1_m, q10, dq1, mo2_w, mo2_ga, mo2_m, q20, dq2, ntot\
                  , dq1_eq=dq1_eq, flag_1PN= False, flag_2_5PN= False)

    xlist = [x for x in range(ntot)]

    plt.figure()
    # plt.style.use('science')
    plt.plot(xlist, an, 'k-')
    plt.title('Iterative map $e$ = '+str(e0))
    plt.xlabel(r'$n$',fontsize=15)
    plt.ylabel(r'$a_n$',fontsize=15)
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()

    return

def tutorial_class():
    from src.util.constants import msun
    import matplotlib.pyplot as plt

    ntot=1500

    m1, r1 = 0.4*msun, 1.e9
    m2, r2 = 0.6*msun, 6.e8
    a0, e0, ga0 = 2.e9, 0.95, 0.

    itsol = itmap()
    itsol.par = {'m1':m1, 'r1':r1, 'm2':m2, 'r2':r2\
                ,'a0':a0, 'e0':e0, 'ga0':ga0}
    itsol.get_mode_1(w=[0.2],olap=[0.5],ell=[2],m=[0],ga=[0.])
    itsol.get_mode_2(w=[],olap=[],ell=[],m=[],ga=[])
    
    itsol.map(ntot)

    xlist = [x for x in range(ntot)]

    plt.figure()
    plt.plot(xlist, itsol.E1n, 'k-')
    plt.title('Iterative map $e$ = '+str(e0))
    plt.xlabel(r'$n$',fontsize=15)
    plt.ylabel(r'$E_{1, n}$',fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()

    return

if __name__ == '__main__':
    # tutorial()
    tutorial_class()