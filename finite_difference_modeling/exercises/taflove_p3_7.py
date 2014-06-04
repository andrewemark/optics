import numpy as np
import matplotlib.pyplot as plt
import matplotlib

""" 
Problem 3.7 from Taflove and Hagness - Computational Electrodynamics 

Solves Maxwell's Equations using a uniform Yee grid for a Gaussian pulse
incident on PEC boundaries. 

The grid is 2D and this models the TMz mode. PEC behaviour is modeled by 
forcing Ez = 0 (Etan = 0) at the boundaries.
"""

c = 3e8 # m/s
dx = 1e4
dt = dx/(c*np.sqrt(2))
eps_0 = 8.854e-12
mu_0 = 4*np.pi*10**(-7)
A = dt/dx

grid_size = 201
grid_midpt = int(np.floor(grid_size/2) + 1)

Ez_Np12 = np.zeros((grid_size, grid_size))
Ez_Nm12 = np.zeros((grid_size, grid_size))
Hx_Np1 = np.zeros((grid_size, grid_size-1))
Hx_N = np.zeros((grid_size, grid_size-1))
Hy_Np1 = np.zeros((grid_size-1, grid_size))
Hy_N = np.zeros((grid_size-1, grid_size))

# Driving Function
F = np.arange(0,60)
FWHM = 20.0 # desired FWHM for gaussian pulse
sigma = FWHM/2.3548
F = np.exp(-(F-30)**2/(2*sigma**2))

# Yee finite difference calculation
for i in range(150):
    Ez_Nm12 = np.copy(Ez_Np12)
    Hx_N = np.copy(Hx_Np1)
    Hy_N = np.copy(Hy_Np1)

    diffs = (Hy_N[1:-1,1:-2] - Hy_N[0:-2,1:-2] + Hx_N[1:-2,0:-2] - Hx_N[1:-2,1:-1])
    Ez_Np12[1:-2,1:-2] = Ez_Nm12[1:-2,1:-2] + (A/eps_0) * diffs
    if i<len(F):
        Ez_Np12[grid_midpt,grid_midpt] = F[i]
    else: 
        Ez_Np12[grid_midpt,grid_midpt] = 0.0
    
    Hx_Np1[:,:] = Hx_N[:,:] + (A/mu_0) * (Ez_Np12[:,:-1] - Ez_Np12[:,1:])
    Hy_Np1[:,:] = Hy_N[:,:] + (A/mu_0) * (Ez_Np12[1:,:]  - Ez_Np12[:-1,:])

plt.pcolor(Ez_Np12)
plt.colorbar()
plt.axis([0,grid_size,0,grid_size])
plt.show()
