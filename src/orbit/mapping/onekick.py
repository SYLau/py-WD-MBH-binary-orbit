import numpy as np
from scipy.integrate import quad
from scipy.special import sinc

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G
from src.util.series import infsum
from src.util.specialcoef import Wlm
from src.orbit.mapping.Hansen import Hansen_Tisserand

def __integrand(u,ell,m,k,e):
    f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))
    return np.cos(k*(u-e*np.sin(u))-m*f)/(1-e*np.cos(u))**ell

def onekick_num(par):
    
    a, e, ga = par['a'], par['e'], par['ga']
    m1,r1,m2,r2 = par['m1'], par['r1'], par['m2'], par['r2']
    mo = par['mo']
    
    Omega = np.sqrt(G*(m1+m2)/a**3)
    kr = mo['w']/Omega

    if mo['role'] == 'primary':
        ep = mo['Wlm']*mo['olap']*(m2/m1)*(r1/a)**(mo['ell']+1)*np.exp(-1j*mo['m']*ga)
    elif mo['role'] == 'seconadry':
        ep = mo['Wlm']*mo['olap']*(m1/m2)*(r2/a)**(mo['ell']+1)*np.exp(-1j*mo['m']*(ga+np.pi))
    else:
        exit('mo[role] not recognized')
    
    res = 2*quad(__integrand,0.,np.pi,args=(mo['ell'],mo['m'],kr,e), epsabs =1.e-14, epsrel =1.e-14)[0]
    return 1j*kr*ep*res

def __terms(k ,par):
    ell, m, w = par['mo']['ell'], par['mo']['m'], par['mo']['w']
    a, e = par['a'], par['e']
    m1, m2 = par['m1'], par['m2']
    P = 2*np.pi*np.sqrt(a**3/G/(m1+m2))
    return Hansen_Tisserand(ell, m, k, e)*sinc(w*P/2/np.pi-k)

def __terms_2(k ,par):
   return __terms(2*k,par) + __terms(2*k+1,par)

def spe_terms(k, par):
    return np.array([__terms(k[i] ,par) for i in range(len(k))])

def spe_terms_2(k, par):
    return np.array([__terms_2(k[i] ,par) for i in range(len(k))])

def onekick_spe(par):
    a, ga = par['a'], par['ga']
    m1,r1,m2,r2 = par['m1'], par['r1'], par['m2'], par['r2']
    mo = par['mo']

    P = 2*np.pi*np.sqrt(a**3/G/(m1+m2))
    w=mo['w']

    if mo['role'] == 'primary':
        ep = mo['Wlm']*mo['olap']*(m2/m1)*(r1/a)**(mo['ell']+1)*np.exp(-1j*mo['m']*ga)
    elif mo['role'] == 'seconadry':
        ep = mo['Wlm']*mo['olap']*(m1/m2)*(r2/a)**(mo['ell']+1)*np.exp(-1j*mo['m']*(ga+np.pi))
    else:
        exit('mo[role] not recognized')
    
    res = infsum(lambda k: __terms_2(k, par), eps=1.e-10, n0=int(w*P/2/np.pi))
    
    return 1j*w*P*ep*res

def asymp():
    # Following Lai 1997
    I2m = {-2: float, 0: float, 2: float}
    # I2m[-2] = lambda z, d: 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z)) \
    #     + d*2*np.sqrt(np.pi)/3*z**2.5*np.exp(-2*z/3)*(-3./10)
    # I2m[0] = lambda z, d: np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z)) \
    #     + d * np.sqrt(np.pi)/4*z**1.5*np.exp(-2*z/3)*(-3./10)
    # I2m[2] = lambda z, d: np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-17./12/z) \
    #     + d*np.sqrt(np.pi)/32*np.sqrt(z)*np.exp(-2*z/3)*(-3./10)
    I2m[-2] = lambda z, d: 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z)) \
        + d*2*np.sqrt(np.pi)/3*z**2.5*np.exp(-2*z/3)*(-3./10+3*np.sqrt(np.pi)/40/np.sqrt(z))
    I2m[0] = lambda z, d: np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z)) \
        + d * np.sqrt(np.pi)/4*z**1.5*np.exp(-2*z/3)*(-3./10 - 3*np.sqrt(np.pi)/20/np.sqrt(z))
    I2m[2] = lambda z, d: np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-17./12/z) \
        + d*np.sqrt(np.pi)/32*np.sqrt(z)*np.exp(-2*z/3)*(-3./10-6./5/z)
    return I2m


def tutorial():
    from src.util.constants import msun
    
    w, olap = 0.2, 0.5
    ell, m = 2., 2.
    mo = {'w':w, 'olap':olap, 'ell':ell, 'm':m, 'Wlm':Wlm(ell,m), 'role':'primary'}
    
    m1, r1 = 0.4*msun, 1.e9
    m2, r2 = 0.6*msun, 6.e8
    a, e, ga = 5.e9, 0.5, 0.
    par = {'a':a, 'e':e, 'ga':ga,\
            'm1':m1, 'r1':r1, 'm2':m2, 'r2':r2,\
            'mo':mo}
    
    num_res = onekick_num(par)
    spe_res = onekick_spe(par)
    
    print(num_res)
    print(spe_res)
    
    klist = np.arange(1,300)
    terms_spe = spe_terms(klist, par)
    print(terms_spe)

if __name__ == '__main__':
    tutorial()
    
    
# class mode_info:
#     def __init__(self,w,olap,ell,m,role='primary'):
#         self.w = w
#         self.olap = olap
#         self.ell = ell
#         self.m = m
#         self.role = role
#         self.cal_Wlm()
    
#     def fact(self,r):
#         n= int(r)
#         if n < 2:
#             return 1
#         else:
#             return n * self.fact(n-1)
    
#     def cal_Wlm(self):
#         ell = self.ell
#         m = self.m
#         lm_p=ell+m
#         lm_m=ell-m
#         if int(lm_p)%2 != 0:
#             self.Wlm = 0.
#         else:
#             self.Wlm = (-1.)**(lm_p/2)*np.sqrt(4*np.pi/(2*ell+1)*self.fact(lm_p)* self.fact(lm_m)) \
#                 /(2**ell*self.fact(lm_p/2)*self.fact(lm_m/2))