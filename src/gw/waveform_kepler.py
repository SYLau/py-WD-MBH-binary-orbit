import numpy as np
from scipy.optimize import newton

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(".")) # put the path to the top directory inside append
from src.util.constants import G, c

class waveform_K:
    
    def __init__(self,par):
        self.par = par

        self.t = np.array([])
        self.r = np.array([])
        self.dr = np.array([])
        self.p = np.array([])
        self.dp = np.array([])
        self.hp = np.array([])
        self.hc = np.array([])
        return
    
    def generate(self,t):
        
        mt,a,e,ga,tp = self.par['mt'],self.par['a'],self.par['e'],self.par['ga'],self.par['tp']

        W = np.sqrt(G*mt/a**3)
        mm = W*(t-tp)
        
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

        self._cal_h()
        return 

    def _cal_h(self):
        mt, mu, L = self.par['mt'], self.par['mu'], self.par['L']
        r,dr,p,dp = self.r, self.dr, self.p, self.dp
        ddr = self.r*self.dp**2 - G*mt/self.r**2
        ddp= -2*self.dr*self.dp/self.r
        self.hp = 2*G*mu/c**4/L*(dr**2*np.cos(2*p) + r*ddr*np.cos(2*p) - 4*r*dr*dp*np.sin(2*p) - 2*r**2*dp**2*np.cos(2*p)-r**2*ddp*np.sin(2*p))
        self.hc = 2*G*mu/c**4/L*(dr**2*np.sin(2*p) + r*ddr*np.sin(2*p) + 4*r*dr*dp*np.cos(2*p) - 2*r**2*dp**2*np.sin(2*p)+r**2*ddp*np.cos(2*p))

        return
    

def tutorial():
    from src.util.constants import msun, Gpc
    import matplotlib.pyplot as plt
    
    mt = 1.2*msun
    mu = 0.3*msun
    a = 1e9
    e = 0.99
    ga = 0.
    L = 1.e-6*Gpc
    par = {'mt':mt, 'mu':mu, 'a':a, 'e':e, 'ga':ga, 'L':L}

    ntot = 2**14
    
    # Using the class waveform
    P0 = 2*np.pi*np.sqrt(a**3/G/mt)
    teval = np.linspace(-P0/2, P0*3/2,ntot)
    gw = waveform_K(par)
    gw.generate(teval)
    
    plt.figure(figsize=(10.,6.4), dpi= 100)
    plt.plot(gw.t,gw.hp,linewidth=1, linestyle = 'solid', color='black')
    plt.xlabel(r'$t$ (s)',fontsize=20)
    plt.ylabel(r'$h_+$',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.close()


if __name__ == '__main__':
    tutorial()