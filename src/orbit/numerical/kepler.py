import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(".")) # put the path to the top directory inside append
from src.util.constants import G

class kepler_num:
    def __init__(self,tend,par):
        self.integrate(tend,par)
        self.f0 = -np.pi
    
    # @staticmethod
    def deriv(self,r,y,mt):
       
        r=y[0]      #r
        dr=y[1]     #dr/dt
        p=y[2]      #phi
        dp=y[3]     #dphi/dt
        
        dydr=0.0*y
        dydr[0]= dr
        dydr[1]= r*dp**2 - G*mt/r**2
        dydr[2]= dp
        dydr[3]= -2*dr*dp/r
        return dydr
    
    def integrate(self,tend,par):
        
        mt,a0,e0,ga0= par['mt'],par['a'],par['e'],par['ga']
        h = np.sqrt(G*mt*a0*(1-e0**2))
        
        y = np.zeros(4)
        y[0] = a0*(1-e0**2)/(1+e0*np.cos(self.f0))
        y[1] = np.sqrt(G*mt/a0/(1-e0**2))*e0*np.sin(self.f0)
        y[2] = ga0 + self.f0
        y[3] = np.sqrt(G*mt/a0**3/(1-e0**2)**3)*(1+e0*np.cos(self.f0))**2
            
        isol = solve_ivp(self.deriv, args = (mt,), t_span=[0,tend], y0 = y,
                        method = 'LSODA', rtol = 1e-10)

        self.t = isol.t
        self.r = isol.y[0,:]
        self.dr = isol.y[1,:]
        self.p = isol.y[2,:]
        self.dp = isol.y[3,:]

        self.x = np.multiply(self.r, np.cos(self.p))
        self.dx = np.multiply(self.dr, np.cos(self.p)) - np.multiply(self.dp, np.multiply(self.r, np.sin(self.p)))
        self.y = np.multiply(self.r, np.sin(self.p))
        self.dy = np.multiply(self.dr, np.sin(self.p))  + np.multiply(self.dp, np.multiply(self.r, np.cos(self.p)))
    
    @staticmethod
    def com(mt,r,dr,dp):
        en = dr**2/2 + r**2*dp**2 - G*mt/r
        h = r**2*dp
        return en, h
    
class kepler_analytic:
    
    def __init__(self,t,par):
        self.solution(t,par)
    
    def solution(self,t,par):
        
        mt,a,e,ga = par['mt'],par['a'],par['e'],par['ga']

        W = np.sqrt(G*mt/a**3)
        mm = W*t
        
        func = lambda u: u-e*np.sin(u)-mm # kepler equation
        u = newton(func,rtol=1.e-10,x0=mm,x1=mm*1.1+0.1) # Numerical error can cause the determination of f problematic

        self.t = t
        self.r = a*(1-e*np.cos(u))
        self.dr = W*a*e*np.sin(u)/(1-e*np.cos(u))
        
        f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))
        # f = np.arctan2(np.sqrt(1-e**2)*np.sin(u), np.cos(u)-e)
        f = np.where(f < 0., f+2*np.pi, f)
        self.p =  ga + f + np.trunc(mm/2/np.pi)*2*np.pi 
        self.dp =  W*np.sqrt(1-e**2)/(1-e*np.cos(u))**2

        self.x = np.multiply(self.r, np.cos(self.p))
        self.dx = np.multiply(self.dr, np.cos(self.p)) - np.multiply(self.dp, np.multiply(self.r, np.sin(self.p)))
        self.y = np.multiply(self.r, np.sin(self.p))
        self.dy = np.multiply(self.dr, np.sin(self.p))  + np.multiply(self.dp, np.multiply(self.r, np.cos(self.p)))

        return 

    @staticmethod
    def com(mt,a,e):
        en = -G*mt/2/a
        h = np.sqrt(G*mt*a*(1-e**2))
        return en, h


def tutorial():
    from src.util.constants import G, msun
    import matplotlib.pyplot as plt
    
    mt = 1.2*msun
    a = 1e9
    e = 0.5
    ga = np.pi/4
    par = {'mt':mt, 'a':a, 'e':e, 'ga':ga}
        
    # Using the class kepler_num
    tend = 10*(2*np.pi*np.sqrt(a**3/G/mt))
    nsol = kepler_num(tend,par)
    
    plt.figure()
    plt.plot(nsol.x,nsol.y,'k.',linewidth=3)
    plt.xlabel(r'$x$ (cm)',fontsize=15)
    plt.ylabel(r'$y$ (cm)',fontsize=15)
    plt.axis('scaled')
    plt.show()
    plt.close()
    
    # Using the class kepler_analytic
    teval = nsol.t
    asol = kepler_analytic(teval,par)
    
    plt.figure()
    plt.plot(nsol.t,nsol.p,linewidth=2.5, linestyle = 'solid', color='black')
    plt.plot(asol.t,asol.p, linewidth=1, linestyle = 'dashed', color='red')
    plt.xlabel(r'$t$ (s)',fontsize=15)
    plt.ylabel(r'$\phi$',fontsize=15)
    plt.show()
    plt.close()


if __name__ == '__main__':
    tutorial()
  
  
# class eom_kepler:
#     def __init__(self,mt):
#         self.mt = mt
    
#     def deriv(self,r,y):
#         from constants import G
#         mt = self.mt
        
#         r=y[0]
#         dr=y[1]
#         p=y[2]
#         dp=y[3]
        
#         dydr=0.0*y
#         dydr[0]= dr
#         dydr[1]= r*dp**2 - G*mt/r**2
#         dydr[2]= dp
#         dydr[3]= -2*dr*dp/r
#         return dydr

# def solve_kepler(tend,mt,a,e):
#     from constants import G

#     eom=eom_kepler(mt)

#     h = np.sqrt(G*mt*a*(1-e**2))
    
#     y = np.zeros(4)
#     y[0] = a*(1-e)        # r0
#     y[1] = 0.             # dr0
#     y[2] = 0.             # p0
#     y[3] = h/y[0]**2      # dp0
        
#     isol = solve_ivp(eom.deriv, t_span=[0,tend], y0 = y,
#                     method = 'LSODA', rtol = 1e-10)

#     return isol

# class kepler_num:
#     def __init__(self,mt,a,e,ga):
#         self.mt = mt
#         self.a = a
#         self.e = e
#         self.ga = ga

#     def deriv(self,r,y):
#         mt = self.mt
        
#         r=y[0]
#         dr=y[1]
#         p=y[2]
#         dp=y[3]
        
#         dydr=0.0*y
#         dydr[0]= dr
#         dydr[1]= r*dp**2 - G*mt/r**2
#         dydr[2]= dp
#         dydr[3]= -2*dr*dp/r
#         return dydr
    
#     def solve_kepler(self,tend):
#         mt=self.mt
#         a=self.a
#         e=self.e
#         ga=self.ga
#         h = np.sqrt(G*mt*a*(1-e**2))
        
#         y = np.zeros(4)
#         y[0] = a*(1-e)        # r0
#         y[1] = 0.             # dr0
#         y[2] = ga             # p0
#         y[3] = h/y[0]**2      # dp0
            
#         isol = solve_ivp(self.deriv, t_span=[0,tend], y0 = y,
#                         method = 'LSODA', rtol = 1e-10)

#         self.t = isol.t
#         self.r = isol.y[0,:]
#         self.dr = isol.y[1,:]
#         self.p = isol.y[2,:]
#         self.dp = isol.y[3,:]
        
#         return 

# class kepler_analytic:
    
#     def __init__(self,mt,a,e,ga):
#         self.mt = mt
#         self.a = a
#         self.e = e
#         self.ga = ga
    
#     def solution(self,t):
#         mt=self.mt
#         a=self.a
#         e=self.e
#         ga=self.ga
        
#         W = np.sqrt(G*mt/a**3)
#         mm = W*t
        
#         func = lambda u: u-e*np.sin(u)-mm # kepler equation
#         u = newton(func,rtol=1.e-10,x0=mm,x1=mm*1.1+0.1) # Numerical error can cause the determination of f problematic

#         self.t = t
#         self.r = a*(1-e*np.cos(u))
#         self.dr = W*a*e*np.sin(u)/(1-e*np.cos(u))
        
#         f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))
#         # f = np.arctan2(np.sqrt(1-e**2)*np.sin(u), np.cos(u)-e)
#         f = np.where(f < 0., f+2*np.pi, f)
#         self.p =  ga + f + np.trunc(mm/2/np.pi)*2*np.pi 
#         self.dp =  W*np.sqrt(1-e**2)/(1-e*np.cos(u))**2

#         return 

#     def kepler_com(self):
#         mt=self.mt
#         a=self.a
#         e=self.e
        
#         en = -G*mt/2/a
#         h = np.sqrt(G*mt*a*(1-e**2))
#         return en, h