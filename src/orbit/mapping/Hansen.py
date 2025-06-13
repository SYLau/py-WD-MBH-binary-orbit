import numpy as np
from scipy.integrate import quad, solve_ivp, simpson
from scipy.special import binom, jv, hyp2f1
from numba import jit

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.series import infsum

def Hansen_direct(ell,m,k,e,method='Simpson'):
    if method == 'Simpson':
        u = np.linspace(0.,np.pi,1000+int(np.abs(k*1000)))
        f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))           #True anomaly
        int_list = np.cos(k*(u-e*np.sin(u))-m*f)/(1-e*np.cos(u))**ell 
        return 2*simpson(int_list,x=u)/2/np.pi
    elif method == 'Quad':
        def integrand(u,ell,m,k,e):
            f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))        #True anomaly
            return np.cos(k*(u-e*np.sin(u))-m*f)/(1-e*np.cos(u))**ell   
        return 2*quad(integrand,0.,np.pi,args=(ell,m,k,e), epsabs =1.e-10, epsrel =1.e-10)[0]/2/np.pi
    elif method == 'RK45':
        def integrand(u,y,ell,m,k,e):
            f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))        #True anomaly
            return np.cos(k*(u-e*np.sin(u))-m*f)/(1-e*np.cos(u))**ell
        return 2*solve_ivp(integrand,t_span=[0.,np.pi],y0=[0.],args=(ell,m,k,e), method='RK45', rtol=1.e-14).y[0,-1]/2/np.pi
    else:
        exit('err: Hansen_direct method not recognized')
    

def __gbinom(n,k):
    if n >= 0:
        return binom(n,k)
    else:
        return (-1)**k*binom(-n+k-1,k)

def __Hfunc(beta,n,m,k,p):
    if k-p-m >= 0:
        H = (-beta)**(k-p-m)*__gbinom(n+1-m,k-p-m)*hyp2f1(k-p-n-1,-m-n-1,k-p-m+1,beta**2)
    elif k-p-m < 0:
        H = (-beta)**(-k+p+m)*__gbinom(n+1+m,-k+p+m)*hyp2f1(-k+p-n-1,m-n-1,-k+p+m+1,beta**2)
    return H

def Hansen_Tisserand(ell ,m ,k:int ,e):
    n = -ell-1
    beta = (1.-np.sqrt(1-e**2))/e
    f = lambda p: jv(p,k*e)*__Hfunc(beta,n,m,k,p)
    if k != 0:
        return (1+beta**2)**(-n-1)*infsum(f, eps=1.e-12)
    else:
        return (1+beta**2)**(-n-1)*__Hfunc(beta,n,m,k,0)


def tutorial():
    import matplotlib.pyplot as plt
    import scienceplots
    from timeit import default_timer as timer
        
    ell, m, e = 2., 2., 0.9
    # k = np.arange(0, int(5/(1.- e)**1.5))
    k = np.arange(-int(1/(1.- e)**1.5), int(5/(1.- e)**1.5))
    
    Xlmk_d = np.array([])    
    start_d = timer()
    time_d = np.array([])
    for i in range(len(k)):
        Xlmk_d= np.append(Xlmk_d, Hansen_direct(ell, m, k[i], e))
        time_d=np.append(time_d, timer()-start_d)
    
    Xlmk_T = np.array([])
    start_T = timer()
    time_T = np.array([])
    for i in range(len(k)):
        Xlmk_T= np.append(Xlmk_T, Hansen_Tisserand(ell, m, k[i], e))
        time_T=np.append(time_T, timer()-start_T)
    
    rdev = np.abs((Xlmk_d-Xlmk_T)/Xlmk_T)
    
    plt.figure()
    # plt.style.use('science')
    plt.plot(k,rdev,'k.',linewidth=3)
    plt.title('Hansen coefficients with $e$ = '+str(e))
    plt.xlabel(r'$k$',fontsize=15)
    plt.ylabel(r'relative deviations',fontsize=15)
    plt.show()
    plt.close()
    
    plt.figure()
    # plt.style.use('science')
    plt.plot(k,Xlmk_d,linewidth=3, color = 'black', linestyle = 'solid', label='Direct integration')
    plt.plot(k,Xlmk_T,linewidth=3, color = 'red', linestyle = 'dashed', label='Tisserand method')
    plt.title('Hansen coefficients $e$ = '+str(e))
    plt.legend(loc ="upper left", fontsize="15", frameon=True) 
    plt.xlabel(r'$k$',fontsize=15)
    plt.ylabel(r'$X^{\ell,m}_{k}$',fontsize=15)
    plt.yscale("log")
    plt.show()
    plt.close()
    
    plt.figure()
    # plt.style.use('science')
    plt.plot(k,time_d,linewidth=3, color = 'black', linestyle = 'solid', label='Direct integration')
    plt.plot(k,time_T,linewidth=3, color = 'red', linestyle = 'dashed', label='Tisserand method')
    plt.title('Computation time of Hansen coefficients with $e$ = '+str(e))
    plt.legend(loc ="upper left", fontsize="15", frameon=True) 
    plt.xlabel(r'$k$',fontsize=15)
    plt.ylabel(r'time (s)',fontsize=15)
    plt.show()
    plt.close()

if __name__ == '__main__':
    tutorial()

# class Hansen_Tisserand:
#     def __init__(self,ell,m,k,e):
#         self.cal_Hansen(ell,m,k,e)

#     def cal_Hansen(self,ell,m,k,e):
#         return self.Tisserand(self,-ell-1,m,k,e)

#     def Tisserand(self,n,m,k,e):
#         beta = (1.-np.sqrt(1-e**2))/e
#         f = lambda p: jv(p,k*e)*self.H_func(beta,n,m,k,p)
#         return (1+beta**2)**(-n-1)*infsum(f, eps=1.e-12)

        