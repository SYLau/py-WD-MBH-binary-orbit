import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import copy
if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G, c
from src.util.specialcoef import Wlm

class binary_dyn_tide_num_eff:
    def __init__(self):
        self.par = {}
        self.mo1 = {}
        self.mo2 = {}
        self.t0 = 0.                #Initial time
        self.f0 = -np.pi            #Initial true anamoly
        self.odemethod = 'RK45'
        self.atol = None
        self.atol_factor = 1e-8
        self.rtol = 1e-8
        self.tide = True
        self.tide_reaction = True
        self.tide_free_prop = False
        self.pn1 = False
        self.pn2_5 = False
        self.redshift = False
        self.anharmonicity = False
        self.keff1 = 1.
        self.keff2 = 1.
        self.y0 = np.array([])

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
       
        r=y[0]      #r
        dr=y[1]     #dr/dt
        p=y[2]      #phi
        dp=y[3]     #dphi/dt
        
        # m1,r1,m2,r2,mo1,mo2 = par.m1,par.r1,par.m2,par.r2,par.mo1,par.mo2
        m1,r1,m2,r2 = par['m1'],par['r1'],par['m2'],par['r2']
        mt = m1+m2
        
        w1,w2 = mo1['w'],mo2['w']
            
        n1=len(w1)
        n2=len(w2)

        d1c=y[4:4+n1]+y[4+n1:4+2*n1]*1j
        d2c=y[4+2*n1:4+2*n1+n2]+y[4+2*n1+n2:4+2*n1+2*n2]*1j

        # Back-reaction force: follow Hang's notes "EfficientEvolEccSys.pdf"
        if self.tide:
            temp1_0 = (mo1['ell']+1)*(mo1['Wlm'] *mo1['olap'])**2 * (r1/r)**(2*mo1['ell']+1)
            temp2_0 = (mo2['ell']+1)*(mo2['Wlm'] *mo2['olap'])**2 * (r2/r)**(2*mo2['ell']+1)
            gr0 = -2*G*mt/r**2* ((m2/m1)*np.sum(temp1_0)+(m1/m2)*np.sum(temp2_0))

            temp1_1 = temp1_0*mo1['m']*dp/w1
            temp2_1 = temp2_0*mo2['m']*dp/w2
            gr1 = -2*G*mt/r**2* ((m2/m1)*np.sum(temp1_1)+(m1/m2)*np.sum(temp2_1))

            temp1 = (mo1['ell']+1)*mo1['Wlm'] *mo1['olap'] * (r1/r)**mo1['ell']*np.real(d1c*np.exp(1j*(mo1['m']*p-w1*t)))
            temp2 = (mo2['ell']+1)*mo2['Wlm'] *mo2['olap'] * (r2/r)**mo2['ell']*np.real(d2c*np.exp(1j*(mo2['m']*(p+np.pi)-w2*t)))
            gr = -2*G*mt/r**2* (np.sum(temp1)+np.sum(temp2))

            temp1_1 = temp1_0*mo1['m']/w1*dr/r
            temp2_1 = temp2_0*mo2['m']/w2*dr/r
            fp1 = 2*G*mt/r**3* ((m2/m1)*np.sum(temp1_1)+(m1/m2)*np.sum(temp2_1))

            temp1 = mo1['m']*mo1['Wlm'] *mo1['olap'] * (r1/r)**mo1['ell']*np.real(1j*d1c*np.exp(1j*(mo1['m']*p-w1*t)))
            temp2 = mo2['m']*mo2['Wlm'] *mo2['olap'] * (r2/r)**mo2['ell']*np.real(1j*d2c*np.exp(1j*(mo2['m']*(p+np.pi)-w2*t)))
            fp = 2*G*mt/r**3* (np.sum(temp1)+np.sum(temp2))

            ddr_tide = gr0 + gr1 + gr
            ddp_tide = fp1 + fp

            if not self.tide_reaction:
                ddr_tide = 0.
                ddp_tide = 0.
                
        else:
            ddr_tide = 0.
            ddp_tide = 0.

        # 1PN and 2.5PN Kidder 1993 EOM (see Eq (2.2))
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
        if self.tide_free_prop:
            Va1 *= 0.
            Va2 *= 0.
        ## Relativistic corrections to mode frequencies: redshift, frame-dragging
        z1, z2 = 1., 1.
        if self.redshift:
            z1 += -( (m2/mt)**2/2*(dr**2 + (r*dp)**2)+G*m2/r)/c**2
            z2 += -((m1/mt)**2/2*(dr**2 + (r*dp)**2)+G*m1/r)/c**2
        ## Mode amplitudes
        q1c = (1 - 1j/w1*(1j*mo1['m']*dp +(mo1['ell']+1)*dr/r) )*Va1*np.exp(-1j*(mo1['m']*p)) + d1c*np.exp(-1j*w1*t)
        q2c = (1 - 1j/w2*(1j*mo2['m']*dp +(mo2['ell']+1)*dr/r) )*Va2*np.exp(-1j*(mo2['m']*p+np.pi)) + d2c*np.exp(-1j*w2*t)
        ## Non-linear tide (anharmonicity)
        ah1, ah2 = 0., 0.
        if self.anharmonicity:
            ah1 = self.keff1*np.abs(q1c)**2*q1c
            ah2 = self.keff2*np.abs(q2c)**2*q2c

        dd1c = -1j/w1*Va1*np.exp(1j*(w1*t-mo1['m']*p)) \
                *( ((1j*mo1['m']*dp) + (mo1['ell']+1)*(dr/r))**2 \
                    - (1j*mo1['m']*ddp+ (mo1['ell']+1)*(ddr/r) - (mo1['ell']+1)*(dr/r)**2) ) \
                    -1j*w1*(z1-1)*q1c*np.exp(1j*w1*t) + 1j*w1*(z1-1)*Va1*np.exp(1j*(w1*t-mo1['m']*p)) \
                    + 1j*w1*z1*ah1*np.exp(1j*w1*t)
                    ## Equivalent expression for redshift
                    # -(z1-1)*(1j*mo1['m']*dp + (mo1['ell']+1)*(dr/r))*Va1*np.exp(1j*(w1*t-mo1['m']*p)) -1j*(z1-1)*w1*d1c \
                    
        dd2c = -1j/w2*Va2*np.exp(1j*(w2*t-mo2['m']*(p + np.pi))) \
                *( ((1j*mo2['m']*dp) + (mo2['ell']+1)*(dr/r))**2 \
                    - (1j*mo2['m']*ddp+ (mo2['ell']+1)*(ddr/r) - (mo2['ell']+1)*(dr/r)**2) ) \
                    -1j*w2*(z2-1)*q2c*np.exp(1j*w2*t) + 1j*w2*(z2-1)*Va2*np.exp(1j*(w2*t-mo2['m']*(p + np.pi))) \
                    + 1j*w2*z2*ah2*np.exp(1j*w2*t)
                    ## Equivalent expression for redshift
                    # -(z2-1)*(1j*mo2['m']*dp + (mo2['ell']+1)*(dr/r))*Va2*np.exp(1j*(w2*t-mo2['m']*(p + np.pi))) -1j*(z2-1)*w2*d2c \
                    
        dydt=0.0*y
        dydt[0]= dr
        dydt[1]= ddr
        dydt[2]= dp
        dydt[3]= ddp
        dydt[4:4+n1]= dd1c.real
        dydt[4+n1:4+2*n1]= dd1c.imag
        dydt[4+2*n1:4+2*n1+n2]= dd2c.real
        dydt[4+2*n1+n2:4+2*n1+2*n2]= dd2c.imag
        return dydt
    
    def solve(self,tend):
        a0,e0,ga0 = self.par['a0'], self.par['e0'], self.par['ga0']
        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        mt = m1+m2
        mo1, mo2 = self.mo1,self.mo2
        q10, q20 = self.par['q1'], self.par['q2']
        
        n1, n2 = len(q10), len(q20)
        y = np.zeros(4+2*n1+2*n2)

        r0 = a0*(1-e0**2)/(1+e0*np.cos(self.f0))
        dr0 = np.sqrt(G*mt/a0/(1-e0**2))*e0*np.sin(self.f0)
        p0 = ga0 + self.f0
        dp0 = np.sqrt(G*mt/a0**3/(1-e0**2)**3)*(1+e0*np.cos(self.f0))**2

        Ua10 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/r0)**(mo1['ell']+1)*np.exp(-1j*mo1['m']*p0)
        Ua20 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/r0)**(mo2['ell']+1)*np.exp(-1j*mo2['m']*(p0+np.pi))

        d10 = q10 - (1- 1j/mo1['w']*(1j*mo1['m']*dp0+(mo1['ell']+1)*(dr0/r0)))*Ua10
        d20 = q20 - (1- 1j/mo2['w']*(1j*mo2['m']*dp0+(mo2['ell']+1)*(dr0/r0)))*Ua20

        y[0] = r0
        y[1] = dr0
        y[2] = p0
        y[3] = dp0

        n1, n2 = len(q10), len(q20)

        y[4:4+n1]=d10.real
        y[4+n1:4+2*n1]=d10.imag
        y[4+2*n1:4+2*n1+n2]=d20.real
        y[4+2*n1+n2:4+2*n1+2*n2]=d20.imag

        if len(self.y0) != 0:
            if len(self.y0) == len(y):
                y = self.y0

        def peri_crossing(t, y,par,mo1,mo2):
            return y[1]
        peri_crossing.direction = 1

        yscale = 0.*y
        yscale[0] = a0
        yscale[1] = np.sqrt(G*mt/a0)
        yscale[2] = np.pi*2
        yscale[3] = np.sqrt(G*mt/a0**3)
        # yscale[4:4+2*n1+2*n2] = 1.e-3
        yscale[4:4+n1] = np.abs(Ua10)
        yscale[4+n1:4+2*n1] = np.abs(Ua10)
        yscale[4+2*n1:4+2*n1+n2] = np.abs(Ua20)
        yscale[4+2*n1+n2:4+2*n1+2*n2] = np.abs(Ua20)
        if self.atol is None:
            self.atol = self.atol_factor*np.abs(yscale)
        isol = solve_ivp(self.deriv, args = (self.par,self.mo1,self.mo2) \
                        , t_span=[self.t0,tend], y0 = y, method = self.odemethod \
                        , atol = self.atol, rtol = self.rtol \
                        , events = peri_crossing)

        self.t = isol.t
        self.ysol = isol.y
        self.r = isol.y[0,:]
        self.dr = isol.y[1,:]
        self.p = isol.y[2,:]
        self.dp = isol.y[3,:]
        # compute q1, q2
        d1 = isol.y[4:4+n1] + isol.y[4+n1:4+2*n1]*1j
        d2 = isol.y[4+2*n1:4+2*n1+n2] + isol.y[4+2*n1+n2:4+2*n1+2*n2]*1j
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

        # if return_sol:
        #     return isol.t, isol.y

    def find_orbele(self):
       
        m1,m2 = self.par['m1'],self.par['m2']
        # r1,r2 = self.par['r1'],self.par['r2']
        mt = m1+m2
        
        en = self.dr**2/2 + self.r**2*self.dp**2/2 - G*mt/self.r
        h = self.r**2*self.dp
        """ 
        Runge-lenz vector
        """
        Ax= self.dy*h/(G*mt) - self.x/self.r
        Ay= -self.dx*h/(G*mt) - self.y/self.r

        self.a = -G*mt/2/en
        self.e = np.sqrt(1-h**2/G/mt/self.a)
        self.ga = np.arctan2(Ay,Ax)

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
            ## Redshift corrections to mode energy
            z1, z2 = np.ones(len(self.r)), np.ones(len(self.r))
            if self.redshift:
                z1 += -( (m2/mt)**2/2*(self.dr**2 + (self.r*self.dp)**2)+G*m2/self.r)/c**2
                z2 += -((m1/mt)**2/2*(self.dr**2 + (self.r*self.dp)**2)+G*m1/self.r)/c**2
            for i in range(len(self.r)):
                en_mode[i] = G*(m1/m2)*mt/r1*np.sum(np.abs(self.q1[:,i])**2)*z1[i]\
                            + G*(m2/m1)*mt/r2*np.sum(np.abs(self.q2[:,i])**2)*z2[i]
                
                
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

    def find_q_eq(self):
        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        mo1, mo2 = self.mo1,self.mo2

        self.q1_eq, self.q2_eq = 0.*self.q1, 0.*self.q2
        for i in range(len(self.r)):    
            Va1 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/self.r[i])**(mo1['ell']+1)
            self.q1_eq[:,i] = (1 - 1j/mo1['w']*(1j*mo1['m']*self.dp[i] +(mo1['ell']+1)*self.dr[i]/self.r[i]) )*Va1*np.exp(-1j*(mo1['m']*self.p[i]))
            Va2 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/self.r[i])**(mo2['ell']+1)
            self.q2_eq[:,i] = (1 - 1j/mo2['w']*(1j*mo2['m']*self.dp[i] +(mo2['ell']+1)*self.dr[i]/self.r[i]) )*Va2*np.exp(-1j*(mo2['m']*(self.p[i]+np.pi)))

    def find_point(self,element='dr',val=0.,trange=None):
        if element == 'dr':
            list = self.dr
        elif element == 'p':
            list = self.p
        elif element == 't':
            list = self.t
        else:
            print('err: binary_dyn_tide_num_eff.find_point element not found')
        
        idx0, idx1 = 0, len(list)
        if trange != None:
            idx0 = (np.abs(self.t - trange[0])).argmin()
            idx1 = (np.abs(self.t - trange[1])).argmin()
        idx = (np.abs(list[idx0:idx1] - val)).argmin()+idx0

        t0, y0 = self.t[idx], self.ysol[:,idx]

        a0,e0,ga0 = self.par['a0'], self.par['e0'], self.par['ga0']
        m1,r1,m2,r2 = self.par['m1'],self.par['r1'],self.par['m2'],self.par['r2']
        mt = m1+m2
        mo1, mo2 = self.mo1,self.mo2
        q10, q20 = self.par['q1'], self.par['q2']
        r0 = a0*(1-e0**2)/(1+e0*np.cos(self.f0))
        p0 = ga0 + self.f0

        Ua10 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/r0)**(mo1['ell']+1)*np.exp(-1j*mo1['m']*p0)
        Ua20 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/r0)**(mo2['ell']+1)*np.exp(-1j*mo2['m']*(p0+np.pi))

        n1, n2 = len(q10), len(q20)

        yscale = 0.*y0
        yscale[0] = a0
        yscale[1] = np.sqrt(G*mt/a0)
        yscale[2] = np.pi*2
        yscale[3] = np.sqrt(G*mt/a0**3)
        yscale[4:4+n1] = np.abs(Ua10)
        yscale[4+n1:4+2*n1] = np.abs(Ua10)
        yscale[4+2*n1:4+2*n1+n2] = np.abs(Ua20)
        yscale[4+2*n1+n2:4+2*n1+2*n2] = np.abs(Ua20)
        if self.atol is None:
            self.atol = self.atol_factor*np.abs(yscale)

        def _func(tend):
            temp_sol = solve_ivp(self.deriv, args = (self.par,self.mo1,self.mo2) \
                , t_span=[t0,tend], y0 = y0, method = self.odemethod \
                , atol = self.atol, rtol = self.rtol)
            if element == 'dr':
                return temp_sol.y[1,-1] - val
            elif element == 'p':
                return temp_sol.y[2,-1] - val
            else:
                return False

        if element != 't':
            # Use root-inding method to locate the time step unless the desired element is t
            tresult = root_scalar(_func, method = 'secant', x0 = self.t[idx]).root
        else:
            tresult = val
        tsol = solve_ivp(self.deriv, args = (self.par,self.mo1,self.mo2) \
                , t_span=[t0,tresult], y0 = y0, method = self.odemethod \
                , atol = self.atol, rtol = self.rtol)
        self.tendf, self.rf, self.drf, self.pf, self.dpf \
            = tsol.t[-1], tsol.y[0,-1], tsol.y[1,-1], tsol.y[2,-1], tsol.y[3,-1]
        self.yendf = tsol.y[:,-1]

        # compute q1, q2
        d1 = tsol.y[4:4+n1,-1] + tsol.y[4+n1:4+2*n1,-1]*1j
        d2 = tsol.y[4+2*n1:4+2*n1+n2,-1] + tsol.y[4+2*n1+n2:4+2*n1+2*n2,-1]*1j
        self.q1f, self.q2f = 0.*d1, 0.*d2
        Ua1 = (m2/m1)*mo1['Wlm']*mo1['olap']*(r1/self.rf)**(mo1['ell']+1)*np.exp(-1j*mo1['m']*self.pf)
        Ua2 = (m1/m2)*mo2['Wlm']*mo2['olap']*(r2/self.rf)**(mo2['ell']+1)*np.exp(-1j*mo2['m']*(self.pf+np.pi))
        self.q1f[:] = d1[:]*np.exp(-1j*mo1['w']*self.tendf) \
                    +(1- 1j/mo1['w']*(1j*mo1['m']*self.dpf+(mo1['ell']+1)*(self.drf/self.rf)))*Ua1
        self.q2f[:] = d2[:]*np.exp(-1j*mo2['w']*self.tendf) \
                    +(1- 1j/mo2['w']*(1j*mo2['m']*self.dpf+(mo2['ell']+1)*(self.drf/self.rf)))*Ua2


def tutorial():
    from src.util.constants import G, msun
    import matplotlib.pyplot as plt

    nsol = binary_dyn_tide_num_eff()

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
