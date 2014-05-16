import numpy as np
import matplotlib.pyplot as plt

""" 
Problem 2.9 from Taflove and Hagness - Computational Electrodynamics 

Solves the 1D scalar wave equation for a Gaussian pulse incident on a 
lossless interface.

The discontinuity at the interface is modeled as a change in the
Courant number.

"""

c = 3e8 # m/s

# Region one
dt1 = 1e-4 
dx1 = 3e4
A1 = (c*dt1)**2
grid_size = 200

# Region two
dt2 = 1e-4/3.5
dx2 = 3e4
A2 = (c*dt2)**2
r2width = 80 # grid spaces

dt = np.zeros(grid_size)
dx = np.zeros(grid_size)
A = np.zeros(grid_size)

for i in range(0,grid_size):
    if i < ((grid_size)-r2width):
        dt[i] = dt1
        dx[i] = dx1
        A[i] = A1
    else:
        dt[i] = dt2
        dx[i] = dx2
        A[i] = A2

u_Np1 = np.zeros(grid_size + 2) # u at next time step
u_Nm1 = np.zeros(grid_size + 2) # u at previous time step

# Gaussian pulse
F = np.arange(0,60)
FWHM = 20.0 # desired FWHM for gaussian pulse
sigma = FWHM/2.3548
F = np.exp(-(F-30)**2/(2*sigma**2))

# Finite difference calculations
for i in range(180):
    # Very inefficient use of "copy" :)
    u_N = np.copy(u_Np1)
    if i<len(F):
        u_N[1] = F[i]
    B = u_N[2:] - 2 * u_N[1:-1] + u_N[0:-2]
    u_Np1[1:-1] = A * (B/dx**2) + 2*u_N[1:-1] - u_Nm1[1:-1]
    u_Nm1 = np.copy(u_N)

plt.figure()
x = np.arange(0,grid_size)
plt.plot(x,u_N[1:-1])
plt.xlabel('i - Grid Coordinate')
plt. ylabel('Wavefunction u(i)')
plt.show()
