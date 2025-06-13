import numpy as np

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.abspath(".")) # put the path to the top directory inside append

def PSD_LISA(f):
    '''
    From Robson 2019
    My own notes:
    If we don't do averaging over F+, Fx, then multiply this factor by R(f). 
    If the waveform amplitude has not included \sin{60}, further multiply it by 4/3. 
    If the waveform has not included both channels, further divide 2.
    '''
    L = 2.5e11              # LISA designed armlength
    fstar = 19.09e-3        # LISA transfer freq
    pOMS = 1.5e-9**2 * (1+ (2.e-3/f)**4)
    pACC = 3.e-13**2 * (1+ (4.e-4/f)**2)*(1+(f/8.e-3)**4)
    psd = 10./3/L**2*(pOMS + 2*(1+np.cos(f/fstar)**2)*pACC/(2.*np.pi*f)**4)*(1. + 0.6*(f/fstar)**2)
    return psd

def tutorial():
    import matplotlib.pyplot as plt

    flist = np.logspace(-5.,0.,1000)
    psdLISA = PSD_LISA(flist)

    plt.figure(figsize=(10.,6.8), dpi= 100)
    plt.plot(flist,psdLISA,linewidth=1, linestyle = 'solid', color='black')
    plt.xlabel(r'$f$ (Hz)',fontsize=20)
    plt.ylabel(r'PSD',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()

if __name__ == '__main__':
    tutorial()