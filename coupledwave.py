import numpy as np
from numpy import sqrt,exp,conj,pi,abs
import scipy, scipy.integrate
import matplotlib.pyplot as plt

def solve_ivp_example():
    def lotkavolterra(t, z, a, b, c, d):
        x, y = z
        return [a*x - b*x*y, -c*y + d*x*y]
    sol = scipy.integrate.solve_ivp(lotkavolterra, [0, 2], [10+1j, 5+1j], args=(1.5, 1, 3, 1), dense_output=True)
    t = np.linspace(0, 2, 401)
    x,y = sol.sol(t)
    plt.plot(x,y)
    plt.show()

def coupledwaveshg(η,L,dk,P1,P2=0,tol=1e-6,plot=True): # η in %/W/cm², L in mm, input power P1 in mW, dk in µm⁻¹
    g = sqrt(η/100/1000/10**2) # g in units of 1/√mW/mm
    E0s = [sqrt(P1)+0j,sqrt(P2)+0j] # initial fields, units of √W (Eᵢ≡√Pᵢ)
    def ODEs(z,Es):
        E1,E2 = Es
        return [-1j * g * exp(-1j*dk*1000*z) * E2 * conj(E1),
                -1j * g * exp(+1j*dk*1000*z) * E1 * E1 ]
    sol = scipy.integrate.solve_ivp(ODEs, [0,L], E0s, dense_output=True, rtol=tol, atol=tol)
    z = np.linspace(0,L,2001)
    PP1,PP2 = [abs(e*e.conj()) for e in sol.sol(z)]
    if plot:
        plt.plot(z,PP1,'r',label='P1')
        plt.plot(z,PP2,'b',label='P2')
        plt.xlabel('z (mm)')
        plt.ylabel('power (mW)')
        plt.legend()
        plt.show()
    return z,PP1,PP2

def coupledwavesfg(η,L,dk,λ1,λ2,P1=0,P2=0,P3=0,loss1=0,loss2=0,loss3=0,tol=1e-6,plot=True):
    # η in %/W/cm², L in mm, input powers in mW, dk in µm⁻¹, loss in dB/cm
    λ3 = 1/(1/λ1+1/λ2)
    h = sqrt(η/100/1000/10**2) # h in units of 1/√mW/mm
    E0s = [sqrt(P)+0j for P in (P1,P2,P3)] # initial fields, units of √W (Eᵢ≡√Pᵢ)
    α1,α2,α3 = [0.01*loss*np.log(10) for loss in (loss1,loss2,loss3)] # e**-α = 10**(-0.1*loss) → α = 0.1*loss*ln10, extra 0.1 for dB/mm
    def ODEs(z,Es):
        E1,E2,E3 = Es
        return [-1j * h * exp(-1j*dk*1000*z) * E3 * conj(E2) * λ3/λ1 - 0.5*α1*E1,
                -1j * h * exp(-1j*dk*1000*z) * E3 * conj(E1) * λ3/λ2 - 0.5*α2*E2,
                -1j * h * exp(+1j*dk*1000*z) * E1 * E2  - 0.5*α3*E3]
    sol = scipy.integrate.solve_ivp(ODEs, [0,L], E0s, dense_output=True, rtol=tol, atol=tol)
    z = np.linspace(0,L,2001)
    PP1,PP2,PP3 = [abs(e*e.conj()) for e in sol.sol(z)]
    if plot:
        plt.plot(z,PP1,'r',label='P1')
        plt.plot(z,PP2,'b',label='P2')
        plt.plot(z,PP3,'g',label='P3')
        plt.xlabel('z (mm)')
        plt.ylabel('power (mW)')
        plt.legend()
        plt.show()
    return z,PP1,PP2,PP3

def coupledwavethg(ηSHG,ηSFG,dkSHG,dkSFG,L,P1=0,P2=0,P3=0,tol=1e-6,plot=True,save=''): # η in %/W/cm², L in mm, input powers in mW, dk in µm⁻¹
    g,h = [sqrt(η/100/1000/10**2) for η in (ηSHG,ηSFG)] # g,h in units of 1/√mW/mm
    E0s = [sqrt(P)+0j for P in (P1,P2,P3)] # initial fields, units of √W (Eᵢ≡√Pᵢ)
    def ODEs(z,Es):
        E1,E2,E3 = Es
        return [-1j * h * exp(-1j*dkSFG*1000*z) * E3 * conj(E2) * 1/3 +
                -1j * g * exp(-1j*dkSHG*1000*z) * E2 * conj(E1),
                -1j * h * exp(-1j*dkSFG*1000*z) * E3 * conj(E1) * 2/3 +
                -1j * g * exp(+1j*dkSHG*1000*z) * E1 * E1,
                -1j * h * exp(+1j*dkSFG*1000*z) * E1 * E2 ]
    sol = scipy.integrate.solve_ivp(ODEs, [0,L], E0s, dense_output=True, rtol=tol, atol=tol)
    z = np.linspace(0,L,2001)
    PP1,PP2,PP3 = [abs(e*e.conj()) for e in sol.sol(z)]
    if plot:
        plt.plot(z,PP1-P1,'r',label='P1-P1$_0$')
        # plt.plot(z,PP1,'r',label='P1')
        plt.plot(z,PP2,'b',label='P2')
        plt.plot(z,PP3,'g',label='P3')
        plt.xlabel('z (mm)')
        plt.ylabel('power (mW)')
        plt.legend()
        plt.show()
    return z,PP1,PP2,PP3

if __name__ == '__main__':
    # solve_ivp_example()
    coupledwaveshg(η=800,L=10,dk=0,P1=1000) # phase matched SHG
    # coupledwaveshg(η=800,L=20,dk=2*pi*0.01,P1=1000,tol=1e-3) # tolerance too low, total power not conserved
    # coupledwaveshg(η=800,L=2,dk=2*pi*1,P1=1000,tol=1e-14) # power conserved to ~1e-10
    coupledwavesfg(η=3200,L=10,dk=2*pi*0.000,λ1=1064,λ2=532,P1=1500,P2=500,tol=1e-6,plot=1) # phase matched SFG
    # coupledwavesfg(η=3200,L=10,dk=2*pi*0.0002,λ1=1064,λ2=532,P1=1500,P2=500,tol=1e-6,plot=1) # phase mis-matched SFG
    coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.0053*2*pi,dkSFG=-0.0053*2*pi,L=10,P1=1000) # Das2006 simulation
    # coupledwavethg(ηSHG=8,ηSFG=3200,dkSHG=0.00015*2*pi,dkSFG=-0.00015*2*pi,L=40,P1=1000) # backconversion
    # coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.00005*2*pi,dkSFG=-0.00005*2*pi,L=40,P1=1000) # chaotic
    # coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.0053*2*pi,dkSFG=-0.0053*2*pi,L=20,P2=2,P1=1,P3=1000) # DFG
    # coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.0053*2*pi,dkSFG=-0.0053*2*pi,L=10,P1=1000,save='Das2006 simulation')
    # coupledwavesfg(η=50,L=10,dk=0,λ1=1064,λ2=532,P1=1000,P2=0.15,plot=1); coupledwavethg(ηSHG=800,ηSFG=50,dkSHG=0.0053*2*pi,dkSFG=-0.0053*2*pi,L=10,P1=1000) # Das2006 comparison
    # coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.0003*2*pi,dkSFG=-0.0003*2*pi,L=10,P1=1000,save='Das2006 simulation 2')
    # coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.0003*2*pi,dkSFG=-0.0003*2*pi,L=100,P1=10,P3=1000,save='Das2006 simulation 3')
    # coupledwavethg(ηSHG=800,ηSFG=3200,dkSHG=0.001*2*pi,dkSFG=-0.001*2*pi,L=100,P1=1,P3=1000,save='Das2006 simulation 4')
    # reference: Das et al "Direct third harmonic generation due to quadratic cascaded processes in periodically poled crystals" (2006)
    