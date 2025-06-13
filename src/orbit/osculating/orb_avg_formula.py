import numpy as np
from numba import njit

if __name__ == '__main__':
    import sys
    sys.path.append(".") # put the path to the top directory inside append
from src.util.constants import G, c

'''Shift of orbital period from 1PN orbital precession: input (a0, e0) at apocenter'''
@njit
def dP_1pn(m1,m2,a0,e0):
    mt = m1+m2
    eta = m1*m2/mt**2
    dP_P0 = -G*mt/c**2/a0/(1+e0)**2*( (1+e0)*(7*(1-eta)-e0*(2-5*eta))\
                                        +5*(1+2*e0)*(1-np.sqrt(1-e0**2))*(2-eta)/e0**2\
                                        -(10-11*eta)/2 - 5*np.sqrt(1-e0**2)*(2-eta))\
    +G*mt/c**2/a0/2/e0**2*(10*(2-eta)*(1-np.sqrt(1-e0**2))+(2+3*eta)*e0**2)
    return dP_P0

'''Change in redshift factor at 1PN order: input a0 at apocenter'''
@njit
def dz_1pn(m1,m2,a0):
    mt=m1+m2
    Ek = -G*mt/2/a0
    dz1 = Ek/c**2*(m2/mt)*(2+m2/mt)
    return dz1

'''Secular change in orbital period due to 2.5 PN GW RR effect'''
@njit
def dP_2_5pn(m1,m2,a,e):
    P = 2*np.pi*np.sqrt(a**3/G/(m1+m2))
    mc=(m1*m2)**(3./5)/(m1+m2)**(1./5)
    return -192.*np.pi/5*(G*mc/c**3*2*np.pi/P)**(5./3)*(1.+73./24*e**2+37./96*e**4)/(1-e**2)**3.5

'''Secular change in eccentricity due to 2.5 PN GW RR effect'''
@njit
def de_2_5pn(m1,m2,a,e):
    mt=m1+m2
    mu=m1*m2/mt
    return -32./5*(G**3/c**5)*(mt**2*mu)/a**4*(19./6*e+121./96*e**3)/(1-e**2)**2.5


'''Secular change in orbital semi-major axis due to 2.5 PN GW RR effect'''
@njit
def da_2_5pn(m1,m2,a,e):
    mt = m1+m2
    eta = m1*m2/mt**2
    return -64./5*eta*c*(G*mt/c**2/a)**3*(1.+73./24*e**2+37./96*e**4)/(1-e**2)**3.5