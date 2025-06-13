import numpy as np
from scipy.integrate import simpson

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G
from src.util.specialcoef import Wlm
from src.orbit.mapping.Hansen import Hansen_Tisserand

def __integrand(u,ell,m,k,e):
    f = 2*np.arctan2(np.sqrt(1+e)*np.sin(u/2), np.sqrt(1-e)*np.cos(u/2))
    return np.exp(1j*(k*(u-e*np.sin(u))-m*f))/(1-e*np.cos(u))**ell

def q_num(u,par):
    
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
    
    ulist = np.linspace(-np.pi,u,num=1000+int(np.abs(kr*1000)))
    int_list = np.array([__integrand(u, mo['ell'],mo['m'], kr, e) for u in ulist])
    return 1j*kr*ep*simpson(int_list,x = ulist)

def qk_spe(t,k,par, num = 1):
    a, e, ga = par['a'], par['e'], par['ga']
    m1,r1,m2,r2 = par['m1'], par['r1'], par['m2'], par['r2']
    mo = par['mo']
    
    Omega = np.sqrt(G*(m1+m2)/a**3)
    P = 2*np.pi/Omega
    w = mo['w']

    if mo['role'] == 'primary':
        ep = mo['Wlm']*mo['olap']*(m2/m1)*(r1/a)**(mo['ell']+1)*np.exp(-1j*mo['m']*ga)
    elif mo['role'] == 'seconadry':
        ep = mo['Wlm']*mo['olap']*(m1/m2)*(r2/a)**(mo['ell']+1)*np.exp(-1j*mo['m']*(ga+np.pi))
    else:
        exit('mo[role] not recognized')
    
    total = Hansen_Tisserand(mo['ell'],mo['m'],k,e) \
            *(np.exp(1j*(w-k*Omega)*t)-np.exp(-1j*(w-k*Omega)*P/2))/1j/(w-k*Omega)
    ki=k
    sn = np.sign(w/Omega-k)
    if sn == 0:
        sn = 1
    for i in range(num-1):
        ki += sn*(-1)**i*(i+1)
        total += Hansen_Tisserand(mo['ell'],mo['m'],ki,e) \
            *(np.exp(1j*(w-ki*Omega)*t)-np.exp(-1j*(w-ki*Omega)*P/2))/1j/(w-ki*Omega)

    return 1j*w*ep*total


def tutorial():
    from src.util.constants import msun
    import matplotlib.pyplot as plt
    import scienceplots
    
    w, olap = 0.2, 0.5
    ell, m = 2, 0
    mo = {'w':w, 'olap':olap, 'ell':ell, 'm':m, 'Wlm':Wlm(ell,m), 'role':'primary'}
    
    m1, r1 = 0.4*msun, 1.e9
    m2, r2 = 0.6*msun, 6.e8
    a, e, ga = 5.e9, 0.5, 0.
    par = {'a':a, 'e':e, 'ga':ga,\
            'm1':m1, 'r1':r1, 'm2':m2, 'r2':r2,\
            'mo':mo}
    
    Omega = np.sqrt(G*(m1+m2)/a**3)
    k0 = round(w/Omega)
    
    ulist = np.linspace(0.,10*np.pi,num=200)
    tlist = (ulist - e*np.sin(ulist))/Omega
    numres = np.zeros(len(ulist),dtype=np.complex_)
    speres1 = np.zeros(len(ulist),dtype=np.complex_)
    speres2 = np.zeros(len(ulist),dtype=np.complex_)
    speres3 = np.zeros(len(ulist),dtype=np.complex_)
    speres4 = np.zeros(len(ulist),dtype=np.complex_)

    for i in range(len(ulist)):
        numres[i] = q_num(ulist[i],par)
        speres1[i] = qk_spe(tlist[i],k0,par)
        speres2[i] = qk_spe(tlist[i],k0,par, num=2)
        speres3[i] = qk_spe(tlist[i],k0,par, num=3)
        speres4[i] = qk_spe(tlist[i],k0,par, num=5)
    

    plt.figure(figsize=(8.6,6.4), dpi= 100)
    plt.style.use('science')
    plt.title(r'$\Delta q$ for ($\ell$, $m$) = '+str(ell)+', '+str(m)+', and (k0, e) = '+str(k0)+', '+str(e))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(ulist/np.pi,np.abs(numres),linewidth=3, color = 'black', linestyle = 'dashed', label='Numerical')
    plt.plot(ulist/np.pi,np.abs(speres1),linewidth=1, color = 'black', linestyle = 'solid', label='1 term')
    plt.plot(ulist/np.pi,np.abs(speres2),linewidth=1, color = 'red', linestyle = 'solid', label='2 terms')
    plt.plot(ulist/np.pi,np.abs(speres3),linewidth=1, color = 'green', linestyle = 'solid', label='3 terms')
    plt.plot(ulist/np.pi,np.abs(speres4),linewidth=1, color = 'blue', linestyle = 'solid', label='5 terms')

    plt.legend(loc ="lower right", fontsize="20", frameon=True) 
    plt.xlabel(r'$u/\pi$',fontsize=20)
    plt.ylabel(r'$\Delta q$',fontsize=20)
    plt.show()
    plt.close()

    exit()

if __name__ == '__main__':
    tutorial()