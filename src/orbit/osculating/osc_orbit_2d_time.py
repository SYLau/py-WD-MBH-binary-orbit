import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append

from src.util.constants import G

class osculating_2d_t:
    def __init__(self):
        self.par = {}
        self.pforce = None
        self.atol_factor = 1.e-10
        self.rtol = 1.e-10
        self.f0 = -np.pi
        self.y0 = np.array([])
        self.extra_eq = None
        self.extra_y0 = np.array([])
        self.extra_yscale = np.array([])
    
    def eq(self, t,y):
        p = y[0]
        e = y[1]
        w = y[2]
        f = y[3]

        m1, m2 = self.par['m1'], self.par['m2']
        mt = m1+m2
        eta = m1*m2/mt**2
        cf = np.cos(f)
        sf = np.sin(f)
        term1 = 1 + e*cf

        R,S = self.pforce(p,e,w,f,self.par)
        
        dpdt = 2*np.sqrt(p**3/G/mt)/term1*S
        dedt = np.sqrt(p/G/mt)*(sf*R + (2*cf+e*(1+cf**2))/term1*S )
        dwdt = np.sqrt(p/G/mt)/e*(-cf*R + (2+e*cf)/term1*sf*S)
        dfdt = np.sqrt(G*mt/p**3)*term1**2 +np.sqrt(p/G/mt)/e*(cf*R - (2+e*cf)/term1*sf*S)
    
        dydt = np.array([dpdt,dedt,dwdt,dfdt])

        if self.extra_eq != None:
            dydt_extra = self.extra_eq(t,y)
            dydt = np.append(dydt,dydt_extra)
            if len(dydt_extra) != len(y)-4:
                exit('err: dydt_extra not the same length as y_extra')

        return dydt
    
    def solve(self,t0,tend):
        
        p0,e0,ga0 = self.par['p0'],self.par['e0'],self.par['ga0']
        # m1, m2 = self.par['m1'], self.par['m2']
        
        y = self.y0
        y = np.zeros(4)
        y[0] = p0
        y[1] = e0
        y[2] = ga0
        y[3] = self.f0
        if len(self.y0) != 0:
            if len(self.y0) == len(y):
                y = self.y0

        yscale = 0.*y
        yscale[0] = p0
        yscale[1] = 1.0
        yscale[2] = np.pi*2
        yscale[3] = np.pi*2

        if self.extra_eq != None:
            y = np.append(y,self.extra_y0)
            yscale = np.append(yscale,self.extra_yscale)
            if len(yscale) != len(y):
                exit('err: y_extra not the same length as y_scale_extra')

        atol = self.atol_factor*yscale
        rtol = self.rtol

        isol = solve_ivp(self.eq, t_span=[t0,tend], y0 = y \
                        , atol = atol, rtol = rtol)
        
        # self.tf = isol.t[-1]
        # self.yf = isol.y[0:4,-1]
        # if self.extra_eq != None:
        #     self.extra_yf = isol.y[4:,-1]

        self.t = isol.t
        self.p = isol.y[0,:]
        self.e = isol.y[1,:]
        self.ga = isol.y[2,:]
        self.f = isol.y[3,:]

        self.ysol = isol.y[0:4,:]

        if self.extra_eq != None:
            self.extra_y = isol.y[4:,:]
        
    
    def find_orbit(self):
        m1, m2 = self.par['m1'], self.par['m2']
        self.r_coord = self.p/(1+self.e*np.cos(self.f))
        self.dr_coord = np.sqrt(G*(m1+m2)/self.p)*self.e*np.sin(self.f)
        self.p_coord = self.f + self.ga
        self.dp_coord = np.sqrt(G*(m1+m2)/self.p**3)*(1+self.e*np.cos(self.f))**2

    def find_point(self,element='f',val=np.pi,trange=None):
        if element == 'p':
            list = self.p
        elif element == 'e':
            list = self.e
        elif element == 'f':
            list = self.f
        elif element == 't':
            list = self.t
        else:
            print('err: osculating_2d_t.find_point element not found')
        
        idx0, idx1 = 0, len(list)
        if trange != None:
            idx0 = (np.abs(self.t - trange[0])).argmin()
            idx1 = (np.abs(self.t - trange[1])).argmin()
        idx = (np.abs(list[idx0:idx1] - val)).argmin()+idx0

        t0, y0 = self.t[idx], self.ysol[:,idx]
        if self.extra_eq != None:
            extra_y0 = self.extra_y[:,idx]

        p0 = self.par['p0']
        yscale = 0.*y0
        yscale[0] = p0
        yscale[1] = 1.0
        yscale[2] = np.pi*2
        yscale[3] = np.pi*2

        if self.extra_eq != None:
            y0 = np.append(y0,extra_y0)
            yscale = np.append(yscale,self.extra_yscale)
            if len(yscale) != len(y0):
                exit('err: y_extra not the same length as y_scale_extra')

        atol = self.atol_factor*yscale
        rtol = self.rtol

        # Solve for the desired point in one integration step
        def _func(tend):
            temp_sol = solve_ivp(self.eq, t_span=[t0,tend], y0 = y0 \
                        , atol = atol, rtol = rtol)
            if element == 'p':
                return temp_sol.y[0,-1] - val
            elif element == 'e':
                return temp_sol.y[1,-1] - val
            elif element == 'f':
                return temp_sol.y[3,-1] - val
            else:
                return False
        if element != 't':
            # Use root-inding method to locate the time step unless the desired element is t
            tresult = root_scalar(_func, method = 'secant', x0 = self.t[idx]).root
        else:
            tresult = val
        tsol = solve_ivp(self.eq, t_span=[t0,tresult], y0 = y0 \
                        , atol = atol, rtol = rtol)

        self.tendf, self.pf, self.ef, self.gaf, self.ff \
            = tsol.t[-1], tsol.y[0,-1], tsol.y[1,-1], tsol.y[2,-1], tsol.y[3,-1]
        if self.extra_eq != None:
            self.extra_yendf = tsol.y[4:,-1]

    # def find_point_2(self,element='f',val=np.pi,trange=None):
        
    #     if element == 'p':
    #         list = self.p
    #     elif element == 'e':
    #         list = self.e
    #     elif element == 'f':
    #         list = self.f
    #         # for i in range(len(list)):
    #         #     if list[i] < 0:
    #         #         list[i] += 2*np.pi # Shift branch cut to zero
    #     elif element == 't':
    #         list = self.t
    #     else:
    #         print('err: osculating_2d_t.find_point element not found')
        
    #     idx0, idx1 = 0, len(list)
    #     if trange != None:
    #         idx0 = (np.abs(self.t - trange[0])).argmin()
    #         idx1 = (np.abs(self.t - trange[1])).argmin()
    #     idx = (np.abs(list[idx0:idx1] - val)).argmin()+idx0

    #     temp = {}
    #     temp['y0'] = self.y0
    #     # temp['tf'] = self.tf
    #     # temp['yf'] = self.yf
    #     t0 = self.t[idx]                      # Replace initial condition with the closest point to val in list
    #     self.y0 = self.ysol[:,idx]
        
    #     if self.extra_eq != None:
    #         temp['extra_y0'] = self.extra_y0
    #         # temp['extra_yf'] = self.extra_yf
    #         self.extra_y0 = self.extra_y[:,idx]
    #         # self.extra_yf = self.extra_yf
        
    #     # Solve for the desired point in one integration step
    #     def _func(tend):
    #         tf, yf, extra_yf = self.solve(t0, tend,save_sol=False, return_sol=True)
    #         if element == 'p':
    #             return yf[0,-1] - val
    #         elif element == 'e':
    #             return yf[1,-1] - val
    #         elif element == 'f':
    #             return yf[3,-1] - val
                
    #         else:
    #             return False
    #     if element != 't':
    #         # Use root-inding method to locate the time step unless the desired element is t
    #         tresult = root_scalar(_func, method = 'secant', x0 = self.t[idx]).root
    #     else:
    #         tresult = val
    #     tf, yf, extra_yf = self.solve(t0, tresult,save_sol=False, return_sol=True)
    #     self.tendf, self.pf, self.ef, self.gaf, self.ff = tf[-1], yf[0,-1], yf[1,-1], yf[2,-1], yf[3,-1]
    #     if self.extra_eq != None:
    #         self.extra_yendf = extra_yf[:,-1]

    #     self.y0 = temp['y0']
    #     # self.tf = temp['tf']
    #     # self.yf = temp['yf']
        
    #     if self.extra_eq != None:
    #         self.extra_y0 = temp['extra_y0']
    #         # self.extra_yf = temp['extra_yf']
    #     return

def tutorial():
    import matplotlib.pyplot as plt
    from scipy import interpolate
    from src.util.constants import c, msun

    # m1 = 1.3*msun
    # m2 = 1.3*msun
    # a0, e0, ga0 = 1.e9/(1-0.9), 0.9, 0.
    m1 = 0.5*msun
    m2 = 1.e5*msun
    a0, e0, ga0 = 4143984437959.14, 0.97, 0.

    eta = m1*m2/(m1+m2)**2

    r0 = a0*(1+e0)
    dr0 = 0.
    dp0 = np.sqrt(G*(m1+m2)/a0**3/(1-e0**2)**3)*(1-e0)**2
    v2 = dr0**2 + r0**2*dp0**2

    h1pn = r0**2*dp0*(1+ ( (1-3*eta)/2*v2 + (3+eta)*G*(m1+m2)/r0)/c**2)
    E1pn = -G*(m1+m2)/2/a0 + (3./8*(1-3*eta)*v2**2 + G*(m1+m2)/2/r0*((3+eta)*v2+G*(m1+m2)/r0))/c**2
    p_1pn = h1pn**2/G/(m1+m2)
    e_1pn = np.sqrt(1+2*p_1pn/G/(m1+m2)*E1pn)

    par = {'m1':m1,'m2':m2,'p0':a0*(1-e0**2),'e0':e0,'ga0':ga0}

    def pforce(p,e,w,f,par_in):
        m1, m2 = par_in['m1'], par_in['m2']
        mt = m1+m2
        eta = m1*m2/mt**2
        cf = np.cos(f)
        sf = np.sin(f)
        term1 = 1 + e*cf

        R = (G*mt)**2/c**2/p**3*term1**2*(-(1+3*eta)*(1+e**2+2*e*cf) \
            + (8-eta)/2*e**2*sf**2 + 2*(2+eta)*term1)
        S = (G*mt)**2/c**2/p**3*term1**2*(2*(2-eta)*e*sf*term1)
        return R,S
    
    osct = osculating_2d_t()
    osct.par = par
    osct.pforce = pforce

    P0k = 2*np.pi*np.sqrt(a0**3/G/(m1+m2))
    osct.solve(0.,1.5*P0k)

    idx1 = (np.abs(osct.t - 0.8*P0k)).argmin()
    idx2 = (np.abs(osct.t - 1.2*P0k)).argmin()
    time = interpolate.interp1d(osct.f[idx1:idx2],osct.t[idx1:idx2])

    print(a0*(1-e0**2),p_1pn)
    print(e0,e_1pn)

    plt.figure()
    plt.plot(osct.t,osct.f,'k.',linewidth=3)
    plt.xlabel(r'$t$ (s)',fontsize=15)
    plt.ylabel(r'$f$',fontsize=15)
    plt.show()
    plt.close()

    # print('time for apocenter crossing = ', time(np.pi))
    # print('Numerical orbit result = ', 338.277)
    print('time for apocenter crossing = ', time(-np.pi))

    return

if __name__ == '__main__':
    tutorial()





'''Old code'''
# def _osc_eq(t,y,par,pforce):

#     p = y[0]
#     e = y[1]
#     w = y[2]
#     f = y[3]

#     m1, m2 = par['m1'], par['m2']
#     mt = m1+m2
#     eta = m1*m2/mt**2
#     cf = np.cos(f)
#     sf = np.sin(f)
#     term1 = 1 + e*cf

#     # R = (G*mt)**2/c**2/p**3*term1**2*(-(1+3*eta)*(1+e**2+2*e*cf) \
#     #     + (8-eta)/2*e**2*sf**2 + 2*(2+eta)*term1)
#     # S = (G*mt)**2/c**2/p**3*term1**2*(2*(2-eta)*e*sf*term1)
#     R,S = pforce(p,e,w,f,par)
    
#     dpdt = 2*np.sqrt(p**3/G/mt)/term1*S
#     dedt = np.sqrt(p/G/mt)*(sf*R + (2*cf+e*(1+cf**2))/term1*S )
#     dwdt = np.sqrt(p/G/mt)/e*(-cf*R + (2+e*cf)/term1*sf*S)
#     dfdt = np.sqrt(G*mt/p**3)*term1**2 +np.sqrt(p/G/mt)/e*(cf*R - (2+e*cf)/term1*sf*S)
  
#     dydt = np.array([dpdt,dedt,dwdt,dfdt])

#     return dydt

# def integrate_osc_eq_time(t0,tend,par,pforce):
    
#     p0,e0,ga0 = par['p0'],par['e0'],par['ga0']
#     m1, m2 = par['m1'], par['m2']

#     y = np.zeros(4)
#     y[0] = p0
#     y[1] = e0
#     y[2] = ga0
#     y[3] = -np.pi

#     yscale = 0.*y
#     yscale[0] = p0
#     yscale[1] = 1.0
#     yscale[2] = np.pi*2
#     yscale[3] = np.pi*2

#     atol = 1.e-10*yscale
#     rtol = 1.e-10

#     isol = solve_ivp(_osc_eq, args = (par,pforce) \
#                     , t_span=[t0,tend], y0 = y \
#                     , atol = atol, rtol = rtol)
#     tsol = isol.t
#     ysol = isol.y

#     return tsol, ysol


# def tutorial():
#     import matplotlib.pyplot as plt
#     from scipy import interpolate
#     from src.util.constants import c, msun

#     m1 = 1.3*msun
#     m2 = 1.3*msun
#     a0, e0, ga0 = 1.e9/(1-0.9), 0.9, 0.

#     eta = m1*m2/(m1+m2)**2

#     r0 = a0*(1+e0)
#     dr0 = 0.
#     dp0 = np.sqrt(G*(m1+m2)/a0**3/(1-e0**2)**3)*(1-e0)**2
#     v2 = dr0**2 + r0**2*dp0**2

#     h1pn = r0**2*dp0*(1+ ( (1-3*eta)/2*v2 + (3+eta)*G*(m1+m2)/r0)/c**2)
#     E1pn = -G*(m1+m2)/2/a0 + (3./8*(1-3*eta)*v2**2 + G*(m1+m2)/2/r0*((3+eta)*v2+G*(m1+m2)/r0))/c**2
#     p_1pn = h1pn**2/G/(m1+m2)
#     e_1pn = np.sqrt(1+2*p_1pn/G/(m1+m2)*E1pn)
    
#     # par = {'m1':m1,'m2':m2,'p0':p_1pn,'e0':e_1pn,'ga0':ga0}
#     par = {'m1':m1,'m2':m2,'p0':a0*(1-e0**2),'e0':e0,'ga0':ga0}

#     def pforce(p,e,w,f,par_in):
#         m1, m2 = par_in['m1'], par_in['m2']
#         mt = m1+m2
#         eta = m1*m2/mt**2
#         cf = np.cos(f)
#         sf = np.sin(f)
#         term1 = 1 + e*cf

#         R = (G*mt)**2/c**2/p**3*term1**2*(-(1+3*eta)*(1+e**2+2*e*cf) \
#             + (8-eta)/2*e**2*sf**2 + 2*(2+eta)*term1)
#         S = (G*mt)**2/c**2/p**3*term1**2*(2*(2-eta)*e*sf*term1)
#         return R,S

#     P0k = 2*np.pi*np.sqrt(a0**3/G/(m1+m2))
#     tsol, ysol = integrate_osc_eq_time(0.,1.5*P0k,par,pforce)

#     time = interpolate.interp1d(ysol[3,:],tsol)

#     print(a0*(1-e0**2),p_1pn)
#     print(e0,e_1pn)

#     print('time for pericenter crossing = ', time(np.pi))
#     print('Numerical orbit result = ', 338.277)

#     # plt.figure()
#     # plt.plot(tsol,ysol[3,:]-2*np.pi,'k.',linewidth=3)
#     # plt.xlabel(r'$t$ (s)',fontsize=15)
#     # plt.ylabel(r'$f$',fontsize=15)
#     # plt.show()
#     # plt.close()

#     return