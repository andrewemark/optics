import numpy as np
import matplotlib.pyplot as plt
import matplotlib

""" 
Problem 3.3 from Taflove and Hagness - Computational Electrodynamics 

Solves Maxwell's Equations using a uniform Yee grid for a Gaussian pulse
incident on a PEC.

PEC behaviour is modeled by forcing Ez = 0 (Etan = 0) at the right boundary.
"""

c = 3e8 # m/s
dx = 1e4
dt = dx/c
eps_0 = 8.854e-12
mu_0 = 4*np.pi*10**(-7)
A = dt/dx

grid_size = 100 

Ez_Np12 = np.zeros(grid_size)
Hy_N = np.zeros(grid_size-1)
Hy_Np1 = np.zeros(grid_size-1)

# Driving Function
F = np.arange(0,60)
FWHM = 20.0 # desired FWHM for gaussian pulse
sigma = FWHM/2.3548
F = np.exp(-(F-30)**2/(2*sigma**2))

# Yee's finite difference calculation for 
for i in range(120):
    Ez_Nm12 = np.copy(Ez_Np12)
    Hy_N = np.copy(Hy_Np1)     
    # Last spot in Ez array is always 0 (Boundary Condition)
    Ez_Np12[0:-2] = Ez_Nm12[0:-2] + (A/eps_0) * (Hy_N[1:] - Hy_N[0:-1])
    if i<len(F):
        Ez_Np12[0] = F[i] 
    Hy_Np1[1:] = Hy_N[1:] + (A/mu_0) * (Ez_Np12[1:-1] - Ez_Np12[0:-2])
    
plt.figure()
plt.subplot(211)
x = np.arange(grid_size)
plt.plot(x, Ez_Np12)
plt.ylabel('Ez')
plt.ylim((-1.1, 1.1))
plt.subplot(212)
plt.plot(x[:-1], Hy_Np1)
plt.xlabel('i - Grid Coordinate')
plt.ylabel('Hy')
plt.ylim((-2/377., 2/377.))
plt.show()
