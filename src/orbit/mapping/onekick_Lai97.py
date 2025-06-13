import numpy as np
from scipy.integrate import solve_ivp, quad, simpson
from scipy.special import airy

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append


def _fcontour(t,x,z,ell,m):
    f = np.exp(-2*z/3*np.sqrt(1+t**2/3)*(1+4./3*t**2)) \
        *(1.-np.sqrt(1+t**2/3)+1j*t)**(2*m)/(1+(t+1j*np.sqrt(1+t**2/3))**2)**(ell+m)
    return np.array([f.real, f.imag])

def _fcontour2(t,z,ell,m):
    f = np.exp(-2*z/3*np.sqrt(1+t**2/3)*(1+4./3*t**2)) \
        *(1.-np.sqrt(1+t**2/3)+1j*t)**(2*m)/(1+(t+1j*np.sqrt(1+t**2/3))**2)**(ell+m)
    return f.real

def _Ilm_semicir(z,ell,m,eps0):
    def __fsemicir(p,z,ell,m,eps0):
        x = 1j + eps0*np.exp(1j*p)
        f = 1j*eps0*np.exp(1j*p)*np.exp(1j*z*(x+x**3/3))*(1+1j*x)**(2*m)/(1+x**2)**(ell+m)
        return f.real
    p = np.linspace(0.,-np.pi,100+int(np.abs((m+1-ell)*1000)))
    int_list = __fsemicir(p,z,ell,m,eps0)
    return simpson(int_list,x=p)

def Ilm_steepest(z,ell,m,eps):
    x = np.array([0.0, 0.0])
    index = m + 1 -ell
    eps0 = eps/np.sqrt(z)
    if index == 0:
        semicir = -np.exp(-2*z/3)/2**(2*m+1)*np.pi
    else:
        # semicir = ((-1)**index -1)* (1j)**(m-ell) * np.exp(-2*z/3)*eps0**index/2.**(ell+m)/index
        semicir = _Ilm_semicir(z,ell,m,eps0)
    # contour = solve_ivp(_fcontour, t_span=[eps0,10./np.sqrt(z)],y0 = x, args=(z,ell,m), atol = 1.e-14, rtol = 1.e-10).y[0,-1]
    contour = solve_ivp(_fcontour, t_span=[eps0,10./np.sqrt(z)],y0 = x, args=(z,ell,m), atol = 1.e-3*np.exp(-2*z/3), rtol = 1.e-10).y[0,-1]
    # contour = quad(_fcontour2, eps0,10./np.sqrt(z), args=(z,ell,m), epsabs = 1.e-3*np.exp(-2*z/3), epsrel = 1.e-10)[0]
    return contour - semicir.real/2

def Ilm_asymp(ell,m,z):
    """ Original expressions from Lai 1997 """
    if ell == 2:
        if m == -2:
            Ilm = 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))
        elif m == 0:
            Ilm = np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z))
        elif m == 2:
            Ilm = np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-89./48/z)
        else:
            exit('Ilm_asymp: m value not support')
    else:
        exit('Ilm_asymp: ell value not support')
    return Ilm

def Ilm_asymp_new(ell,m,z):
    """ Rederived by me using the steepest descent method """
    if ell == 2:
        if m == -2:
            Ilm = 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))
        elif m == 0:
            Ilm = np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z)+101./144/z)
        elif m == 2:
            Ilm = np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-17./12/z)
        else:
            exit('Ilm_asymp_new: m value not support')
    else:
        exit('Ilm_asymp_new: ell value not support')
    return Ilm

def Ilm_PT_fit(ell,m,z):
    """
        Using the recursion relation from Press and Teukolsky 1977 to find K_{lm}
        The integral is evaluated by assuming a parabolic orbit, which depends on a single parameter y
        Parabolic orbit is described by the Barker's equation 
        Code copied from Fortran project proj_eom_chaos
    """
    return

def tutorial():
    import matplotlib.pyplot as plt
    import scienceplots
    
    ell, m = 2, -2
    kp_list = np.linspace(0.5,20,num=50)
    
    eps = 1.
    
    # I2m_asymp = asymp()
    # myI2m_asymp = asymp_new()
    
    dout = {'Ilm':np.zeros(len(kp_list)), 'IlmLai':np.zeros(len(kp_list)), 'myIlm':np.zeros(len(kp_list))}
    for j in range(len(kp_list)):
        y = np.sqrt(2)*kp_list[j]
        dout['Ilm'][j] = Ilm_steepest(y,ell,-m,eps)
        dout['IlmLai'][j] = Ilm_asymp(ell,-m,y)
        dout['myIlm'][j] = Ilm_asymp_new(ell,-m,y)
    
    plt.figure(figsize=(8.6,6.4), dpi= 100)
    plt.style.use('science')
    plt.title(r'$I_{\ell m}$ for $\ell$, $m$ = '+str(ell)+', '+str(m))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(kp_list,np.abs(dout['Ilm']),linewidth=2, color = 'black', linestyle = 'solid', label='Integral')
    plt.plot(kp_list,dout['IlmLai'],'k.', label='Lai 1997')
    plt.plot(kp_list,dout['myIlm'],'k.', color = 'red', label='my expression')

    plt.legend(loc ="upper right", fontsize="20", frameon=True) 
    plt.xlabel(r'$k_p$',fontsize=20)
    plt.ylabel(r'$I_{\ell m}$',fontsize=20)
    plt.yscale('log')
    plt.show()
    plt.close()

if __name__ == '__main__':
    tutorial()



# def asymp():
#     # Derived by Lai 1997; Obsolete, don't use
#     I2m = {-2: float, 0: float, 2: float}
#     I2m[-2] = lambda z: 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))
#     I2m[0] = lambda z: np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z))
#     I2m[2] = lambda z: np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-89./48/z)
#     return I2m

# def asymp_new():
#     # Rederived by me; Obsolete, don't use
#     I2m = {-2: float, 0: float, 2: float}
#     # I2m[-2] = lambda z: 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z)+5./16/z)
#     I2m[-2] = lambda z: 2*np.sqrt(np.pi)/3*z**1.5*np.exp(-2*z/3)*(1-np.sqrt(np.pi)/4/np.sqrt(z))
#     I2m[0] = lambda z: np.sqrt(np.pi)/4*np.sqrt(z)*np.exp(-2*z/3)*(1+np.sqrt(np.pi)/2/np.sqrt(z)+101./144/z)
#     I2m[2] = lambda z: np.sqrt(np.pi)/32/np.sqrt(z)*np.exp(-2*z/3)*(1-17./12/z)
#     return I2m