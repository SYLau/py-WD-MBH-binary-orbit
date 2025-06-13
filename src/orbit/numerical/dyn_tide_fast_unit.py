import numpy as np
from scipy.integrate import solve_ivp
if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G, c
from src.util.specialcoef import Wlm

class binary_dyn_tide_num_eff_unit:
    def __init__(self):
        self.par = {}
        self.mo1 = {}
        self.mo2 = {}
        self.f0 = -np.pi            #Initial true anamoly
        self.odemethod = 'RK45'
        self.t_unit = 1.
        self.r_unit = 1.
        self.q_unit = 1.
        self.atol = 1e-8
        self.rtol = 1e-8
        self.tide = True
        self.pn1 = False
        self.pn2_5 = False

    @staticmethod
    def gen_mode(w,olap,ell,m):
        nn,nl,nm = len(w), len(ell), len(m)
        if len(olap) != nn:
            exit("Length of w differs from olap")
        mo={'w':np.zeros(nn*nl*nm), 'olap':np.zeros(nn*nl*nm), 'ell':np.zeros(nn*nl*nm), 'm':np.zeros(nn*nl*nm)\
            , 'Wlm':np.zeros(nn*nl*nm)}
        ind = 0
        for i1 in range(nn):
            for i2 in range(nl):
                for i3 in range(nm):
                    mo['w'][ind] = w[i1]
                    mo['olap'][ind] = olap[i1]
                    mo['ell'][ind] = ell[i2]
                    mo['m'][ind] = m[i3]
                    mo['Wlm'][ind] = Wlm(ell[i2],m[i3])
                    ind += 1
        return mo
    
    # @staticmethod
    def deriv(self,t,y,par,mo1,mo2):
        
        ts = t*self.t_unit

        r=y[0]*self.r_unit                  #r
        dr=y[1]*self.r_unit/self.t_unit     #dr/dt
        p=y[2]                              #phi
        dp=y[3]/self.t_unit                 #dphi/dt
        
        # m1,r1,m2,r2,mo1,mo2 = par.m1,par.r1,par.m2,par.r2,par.mo1,par.mo2
        m1,r1,m2,r2 = par['m1'],par['r1'],par['m2'],par['r2']
        mt = m1+m2
        
        n1=len(mo1['w'])
        n2=len(mo2['w'])

        d1c=(y[4:4+n1]+y[4+n1:4+2*n1]*1j)*self.q_unit
        d2c=(y[4+2*n1:4+2*n1+n2]+y[4+2*n1+n2:4+2*n1+2*n2]*1j)*self.q_unit
        
        # Back-reaction force: follow Hang's notes "EfficientEvolEccSys.pdf"
        if self.tide:
            temp1_0 = (mo1['ell']+1)*(mo1['Wlm'] *mo1['olap'])**2 * (r1/r)**(2*mo1['ell']+1)
            temp2_0 = (mo2['ell']+1)*(mo2['Wlm'] *mo2['olap'])**2 * (r2/r)**(2*mo2['ell']+1)
            gr0 = -2*G*mt/r**2* ((m2/m1)*np.sum(temp1_0)+(m1/m2)*np.sum(temp2_0))

            temp1_1 = temp1_0*mo1['m']*dp/mo1['w']
            temp2_1 = temp2_0*mo2['m']*dp/mo2['w']
            gr1 = -2*G*mt/r**2* ((m2/m1)*np.sum(temp1_1)+(m1/m2)*np.sum(temp2_1))

            temp1 = (mo1['ell']+1)*mo1['Wlm'] *mo1['olap'] * (r1/r)**mo1['ell']*np.real(d1c*np.exp(1j*(mo1['m']*p-mo1['w']*ts)))
            temp2 = (mo2['ell']+1)*mo2['Wlm'] *mo2['olap'] * (r2/r)**mo2['ell']*np.real(d2c*np.exp(1j*(mo2['m']*(p+np.pi)-mo2['w']*ts)))
            gr = -2*G*mt/r**2* (np.sum(temp1)+np.sum(temp2))

            temp1_1 = temp1_0*mo1['m']/mo1['w']*dr/r
            temp2_1 = temp2_0*mo2['m']/mo2['w']*dr/r
            fp1 = 2*G*mt/r**3* ((m2/m1)*np.sum(temp1_1)+(m1/m2)*np.sum(temp2_1))

            temp1 = mo1['m']*mo1['Wlm'] *mo1['olap'] * (r1/r)**mo1['ell']*np.real(1j*d1c*np.exp(1j*(mo1['m']*p-mo1['w']*ts)))
            temp2 = mo2['m']*mo2['Wlm'] *mo2['olap'] * (r2/r)**mo2['ell']*np.real(1j*d2c*np.exp(1j*(mo2['m']*(p+np.pi)-mo2['w']*ts)))
            fp = 2*G*mt/r**3* (np.sum(temp1)+np.sum(temp2))

            ddr_tide = gr0 + gr1 + gr
            ddp_tide = fp1 + fp
        else:
            ddr_tide = 0.
            ddp_tide = 0.

        # 1PN and 2.5 PN Kidder 1993 EOM (see Eq (2.2))
        eta = m1*m2/mt**2
        m_r=G*mt/r/c**2
        dr_c=dr/c
        v_c=np.sqrt(dr**2+r**2*dp**2)/c
        if self.pn1:
            a1=-2.*(2.+eta)*m_r+(1.+3.*eta)*v_c**2-1.5*eta*dr_c**2
            b1=-2.*(2.-eta)*dr_c
            ddr_pn1, ddp_pn1 = -G*mt/r**2*(a1+b1*dr_c), -G*mt/c/r**2*b1*dp
        else:
            ddr_pn1, ddp_pn1 = 0., 0.
        if self.pn2_5:
            a2_5=-8./5*eta*m_r*dr_c*(18*v_c**2+2./3*m_r-25*dr_c**2)
            b2_5=8./5*eta*m_r*(6*v_c**2-2*m_r-15*dr_c**2)
            ddr_pn2_5, ddp_pn2_5 = -G*mt/r**2*(a2_5+b2_5*dr_c), -G*mt/c/r**2*b2_5*dp
        else:
            ddr_pn2_5, ddp_pn2_5 = 0., 0.

        ddr = r*dp**2 - G*mt/r**2 + ddr_tide + ddr_pn1 + ddr_pn2_5
        ddp = -2*dr*dp/r + ddp_tide + ddp_pn1 + ddp_pn2_5

        # Mode amplitude derivative in time
        Va1 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/r)**(mo1['ell']+1)
        Va2 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/r)**(mo2['ell']+1)
        dd1c = -1j/mo1['w']*Va1*np.exp(1j*(mo1['w']*ts-mo1['m']*p)) \
                *( ((1j*mo1['m']*dp) + (mo1['ell']+1)*(dr/r))**2 \
                    - (1j*mo1['m']*ddp+ (mo1['ell']+1)*(ddr/r) - (mo1['ell']+1)*(dr/r)**2) )
        dd2c = -1j/mo2['w']*Va2*np.exp(1j*(mo2['w']*ts-mo2['m']*(p + np.pi))) \
                *( ((1j*mo2['m']*dp) + (mo2['ell']+1)*(dr/r))**2 \
                    - (1j*mo2['m']*ddp+ (mo2['ell']+1)*(ddr/r) - (mo2['ell']+1)*(dr/r)**2) )

        dydt=0.0*y
        dydt[0]= dr/(self.r_unit/self.t_unit)
        dydt[1]= ddr/(self.r_unit/self.t_unit**2)
        dydt[2]= dp/(1./self.t_unit)
        dydt[3]= ddp/(1./self.t_unit**2)
        dydt[4:4+n1]= dd1c.real/(self.q_unit/self.t_unit)
        dydt[4+n1:4+2*n1]= dd1c.imag/(self.q_unit/self.t_unit)
        dydt[4+2*n1:4+2*n1+n2]= dd2c.real/(self.q_unit/self.t_unit)
        dydt[4+2*n1+n2:4+2*n1+2*n2]= dd2c.imag/(self.q_unit/self.t_unit)
        return dydt
    
    def solve(self,tend):
        a0,e0,ga0 = self.par['a0'], self.par['e0'], self.par['ga0']
        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        mt = m1+m2
        mo1, mo2 = self.mo1,self.mo2
        q10, q20 = self.par['q1'], self.par['q2']

        # h = np.sqrt(G*mt*a0*(1-e0**2))
        
        n1, n2 = len(q10), len(q20)
        y = np.zeros(4+2*n1+2*n2)

        # r0 = a0*(1+e0)
        # dr0 = 0.
        # p0 = ga0-np.pi
        # dp0 = h/r0**2
        r0 = a0*(1-e0**2)/(1+e0*np.cos(self.f0))
        dr0 = np.sqrt(G*mt/a0/(1-e0**2))*e0*np.sin(self.f0)
        p0 = ga0 + self.f0
        dp0 = np.sqrt(G*mt/a0**3/(1-e0**2)**3)*(1+e0*np.cos(self.f0))**2

        Ua10 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/r0)**(mo1['ell']+1)*np.exp(-1j*mo1['m']*p0)
        Ua20 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/r0)**(mo2['ell']+1)*np.exp(-1j*mo2['m']*(p0+np.pi))

        d10 = q10 - (1- 1j/mo1['w']*(1j*mo1['m']*dp0+(mo1['ell']+1)*(dr0/r0)))*Ua10
        d20 = q20 - (1- 1j/mo2['w']*(1j*mo2['m']*dp0+(mo2['ell']+1)*(dr0/r0)))*Ua20

        y[0] = r0/self.r_unit
        y[1] = dr0/(self.r_unit/self.t_unit)
        y[2] = p0
        y[3] = dp0/(1./self.t_unit)

        n1, n2 = len(q10), len(q20)

        y[4:4+n1]=d10.real/(self.q_unit)
        y[4+n1:4+2*n1]=d10.imag/(self.q_unit)
        y[4+2*n1:4+2*n1+n2]=d20.real/(self.q_unit)
        y[4+2*n1+n2:4+2*n1+2*n2]=d20.imag/(self.q_unit)

        def peri_crossing(t, y,par,mo1,mo2):
            return y[1]
        peri_crossing.direction = 1

        # yscale = 0.*y
        # yscale[0] = a0
        # yscale[1] = np.sqrt(G*mt/a0)
        # yscale[2] = np.pi*2
        # yscale[3] = np.sqrt(G*mt/a0**3)
        # yscale[4:4+n1] = np.abs(Ua10)
        # yscale[4+n1:4+2*n1] = np.abs(Ua10)
        # yscale[4+2*n1:4+2*n1+n2] = np.abs(Ua20)
        # yscale[4+2*n1+n2:4+2*n1+2*n2] = np.abs(Ua20)
        isol = solve_ivp(self.deriv, args = (self.par,self.mo1,self.mo2) \
                        , t_span=[0,tend/self.t_unit], y0 = y, method = self.odemethod \
                        , atol = self.atol, rtol = self.rtol \
                        , events = peri_crossing)

        self.t = isol.t*self.t_unit
        self.r = isol.y[0,:]*self.r_unit
        self.dr = isol.y[1,:]*self.r_unit/self.t_unit
        self.p = isol.y[2,:]
        self.dp = isol.y[3,:]/self.t_unit
        # compute q1, q2
        d1 = (isol.y[4:4+n1] + isol.y[4+n1:4+2*n1]*1j)*self.q_unit
        d2 = (isol.y[4+2*n1:4+2*n1+n2] + isol.y[4+2*n1+n2:4+2*n1+2*n2]*1j)*self.q_unit
        self.q1, self.q2 = 0.*d1, 0.*d2
        for i in range(len(self.r)):
            Ua1 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/self.r[i])**(mo1['ell']+1)*np.exp(-1j*mo1['m']*self.p[i])
            Ua2 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/self.r[i])**(mo2['ell']+1)*np.exp(-1j*mo2['m']*(self.p[i]+np.pi))
            self.q1[:,i] = d1[:,i]*np.exp(-1j*mo1['w']*self.t[i]) \
                        +(1- 1j/mo1['w']*(1j*mo1['m']*self.dp[i]+(mo1['ell']+1)*(self.dr[i]/self.r[i])))*Ua1
            self.q2[:,i] = d2[:,i]*np.exp(-1j*mo2['w']*self.t[i]) \
                        +(1- 1j/mo2['w']*(1j*mo2['m']*self.dp[i]+(mo2['ell']+1)*(self.dr[i]/self.r[i])))*Ua2

        self.x = np.multiply(self.r, np.cos(self.p))
        self.dx = np.multiply(self.dr, np.cos(self.p)) - np.multiply(self.dp, np.multiply(self.r, np.sin(self.p)))
        self.y = np.multiply(self.r, np.sin(self.p))
        self.dy = np.multiply(self.dr, np.sin(self.p))  + np.multiply(self.dp, np.multiply(self.r, np.cos(self.p)))

    def find_orbele(self):
        """ 
        Still working on it.
        Need to include ga, etc
        """
        
        m1,m2 = self.par['m1'],self.par['m2']
        r1,r2 = self.par['r1'],self.par['r2']
        mt = m1+m2
        
        en = self.dr**2/2 + self.r**2*self.dp**2/2 - G*mt/self.r
        h = self.r**2*self.dp
        self.a = -G*mt/2/en
        self.e = np.sqrt(1-h**2/G/mt/self.a)

    def find_energy(self):
        m1,m2 = self.par['m1'],self.par['m2']
        r1,r2 = self.par['r1'],self.par['r2']
        mt = m1+m2
        
        # Interaction energy
        en_int = np.zeros(len(self.r))
        if self.tide:
            for i in range(len(self.r)):
                temp1 = self.mo1['Wlm'] *self.mo1['olap']* (r1/self.r[i])**self.mo1['ell'] \
                    *np.real(self.q1[:,i]*np.exp(1j*self.mo1['m']*self.p[i]))
                temp2 = self.mo2['Wlm'] *self.mo2['olap']* (r2/self.r[i])**self.mo2['ell'] \
                    *np.real(self.q2[:,i]*np.exp(1j*self.mo2['m']*(self.p[i]+np.pi)))
                en_int[i] = -2*G*mt/self.r[i]* (np.sum(temp1)+np.sum(temp2))

        # Mode energy
        en_mode = np.zeros(len(self.r))
        if self.tide:
            for i in range(len(self.r)):
                en_mode[i] = G*(m1/m2)*mt/r1*np.sum(np.abs(self.q1[:,i])**2)\
                    + G*(m2/m1)*mt/r2*np.sum(np.abs(self.q2[:,i])**2)
                
        # 1PN energy
        en_1pn = np.zeros(len(self.r))
        if self.pn1:
            eta = m1*m2/mt**2
            v=np.sqrt(self.dr**2+self.r**2*self.dp**2)
            en_1pn = (3./8*(1-3*eta)*v**4 + G*mt/2/self.r*((3+eta)*v**2+eta*self.dr**2+G*mt/self.r))/c**2
        
        self.en_orb = self.dr**2/2 + self.r**2*self.dp**2/2 - G*mt/self.r
        self.en_mode = en_mode
        self.en_int = en_int
        self.en_1pn = en_1pn
        self.en = self.en_orb + en_int +en_mode +en_1pn


def tutorial():
    from src.util.constants import G, msun
    import matplotlib.pyplot as plt

    nsol = binary_dyn_tide_num_eff_unit()

    # mo1 = nsol.gen_mode(w=np.array([0.2]), olap=np.array([0.5]), ell=np.array([2]), m=np.array([-2, 0, 2]))
    mo1 = nsol.gen_mode(w=np.array([0.5]), olap=np.array([0.5]), ell=np.array([2]), m=np.array([-2, 0, 2]))
    # mo2 = nsol.gen_mode(w=np.array([0.6]), olap=np.array([0.7]), ell=np.array([2]), m=np.array([-2, 0, 2]))
    mo2 = nsol.gen_mode(w=np.array([]), olap=np.array([]), ell=np.array([]), m=np.array([]))

    m1, r1 = 0.4*msun, 1.e9
    m2, r2 = 0.6*msun, 6.e8
    # a, e, ga = 10.e9, 0.5, 0.
    a0, e0, ga0 = 5.5e9, 0.5, 0.
    q1 = np.zeros(len(mo1['w']))
    q2 = np.zeros(len(mo2['w']))
    
    par = {'a0':a0, 'e0':e0, 'ga0':ga0,\
            'm1':m1, 'r1':r1, 'm2':m2, 'r2':r2,\
            'q1':q1, 'q2':q2}
    
    P0K = 2*np.pi*np.sqrt(a0**3/G/(m1+m2))
    # tend = 50*P0K
    tend = 10*P0K
 
    # print('t', tend)
    # print('k', mo1['w'][0]/np.sqrt(G*(m1+m2)/a**3))
    # print('k_p', mo1['w'][0]/np.sqrt(G*(m1+m2)/a**3)*(1-e)**1.5)

    
    nsol.par = par
    nsol.mo1, nsol.mo2 = mo1, mo2
    # nsol.pn1 = True

    nsol.t_unit = 2*np.pi*np.sqrt(a0**3/G/(m1+m2))
    nsol.r_unit = a0
    nsol.q_unit = 1.e-3

    nsol.solve(tend)
    
    nsol.x = np.multiply(nsol.r, np.cos(nsol.p))
    nsol.y = np.multiply(nsol.r, np.sin(nsol.p)) 
    
    plt.figure()
    plt.plot(nsol.x,nsol.y,'k.',linewidth=3)
    plt.xlabel(r'$x$ (cm)',fontsize=15)
    plt.ylabel(r'$y$ (cm)',fontsize=15)
    plt.axis('scaled')
    plt.show()
    plt.close()

    nsol.find_orbele()

    plt.figure()
    plt.plot(nsol.t/P0K,nsol.a,'k.',linewidth=3)
    plt.xlabel(r'$n_K$',fontsize=15)
    plt.ylabel(r'$a$ (cm)',fontsize=15)
    plt.show()
    plt.close()

    # plt.figure()
    # plt.plot(np.real(nsol.q1[0,:]),np.imag(nsol.q1[0,:]),ls='',marker='.')
    # plt.xlabel(r'$R q_1$',fontsize=15)
    # plt.ylabel(r'$I q_1$',fontsize=15)
    # plt.show()
    # plt.close()

if __name__ == '__main__':
    tutorial()
