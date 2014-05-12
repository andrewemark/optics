import numpy as np
from scipy.integrate import quad
from scipy.special import jv

# Computes the optical fields near the focus of a lens for a
# gaussian beam input. It is based on the general formulations 
# set forth in Novotny and Hecht's Principles of Nano-Optics - Chapter 3. 

def cart2pol(x, y):
    """ 
    Cartesian coordinates to polar coordinate conversion for numpy arrays
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Handle singularity for x = 0
    idx = np.where(x==0)
    theta[idx] = 0
    return (r, theta)

def complex_int(func, a, b, **kwargs):
    """
    Performs quadrature integration on complex functions
    """
    def real_func(*argv):
        return np.real(func(*argv))
    def imag_func(*argv):
        return np.imag(func(*argv))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]

# The following integrands are those defined Novotny and Hecht, S3.6

def f_w(t, w0, t_max, f):
    """ Apodization function 
    """
    f0 = w0/(f*np.sin(t_max))
    return np.exp(-1*(np.sin(t)/((f0*np.sin(t_max))))**2)

def I00_integrand(t, f_w, t_max, k, rho, z, w0, f):
    return f_w(t, w0, t_max, f) * \
    (np.cos(t))**(0.5)*np.sin(t)*(1+np.cos(t))*jv(0,k*rho*np.sin(t))*np.exp(1j*k*z*np.cos(t))

def I01_integrand(t, f_w, t_max, k, rho, z, w0, f):
    return f_w(t, w0, t_max, f) * \
    (np.cos(t))**(0.5)*(np.sin(t))**2*jv(1,k*rho*np.sin(t))*np.exp(1j*k*z*np.cos(t))

def I02_integrand(t, f_w, t_max, k, rho, z, w0, f):
    return f_w(t, w0, t_max, f) * \
    (np.cos(t))**(0.5)*np.sin(t)*(1-np.cos(t))*jv(2,k*rho*np.sin(t))*np.exp(1j*k*z*np.cos(t))


# Compute the fields near focus in various planes

def compute_YZ_fields_TEM00(y, z, t_max, k, E0, f, n, w0):
    """
    Returns: matricies of the field components, Ex, Ey, Ez, in the Y, Z plane
    and coordinate mesh
    """
    
    phi = np.pi/2.0
    yy, zz = np.meshgrid(y, z)

    # TODO: Put this into function
    I00 = np.zeros((np.size(yy,0),np.size(zz,1)), dtype=complex)
    I01 = np.zeros((np.size(yy,0),np.size(zz,1)), dtype=complex)
    I02 = np.zeros((np.size(yy,0),np.size(zz,1)), dtype=complex)
    Ex = np.zeros((np.size(yy,0),np.size(zz,1)), dtype=complex)
    Ey = np.zeros((np.size(yy,0),np.size(zz,1)), dtype=complex)
    Ez = np.zeros((np.size(yy,0),np.size(zz,1)), dtype=complex)
    
    E_const = (1j*k*f/2.0) * np.sqrt(1/n) * E0 * np.exp(-1j*k*f)
    
    for i in range(0,np.size(yy,0)):
        for j in range(0, np.size(zz,1)):
            I00[i,j] = complex_int(I00_integrand, 0, t_max, args=(f_w, t_max, k, abs(yy[i,j]), zz[i,j], w0, f))
            I01[i,j] = complex_int(I01_integrand, 0, t_max, args=(f_w, t_max, k, abs(yy[i,j]), zz[i,j], w0, f))
            I02[i,j] = complex_int(I02_integrand, 0, t_max, args=(f_w, t_max, k, abs(yy[i,j]), zz[i,j], w0, f))
            Ex[i,j] = E_const * (I00[i,j]+I02[i,j]*np.cos(2*phi))
            Ey[i,j] = E_const * (I02[i,j]*np.sin(2*phi))
            Ez[i,j] = E_const * (-2j*I01[i,j]*np.cos(phi))

    return (Ex, Ey, Ez, yy, zz)

def compute_XY_fields_TEM00(x, y, t_max, k, E0, f, n, w0):

    # TODO: Allow evaluation in arbitrary z planes! Forced to z = 0 right now.
    xx, yy = np.meshgrid(x, y)
    rho, phi = cart2pol(xx, yy)

    I00 = np.zeros((np.size(xx,0),np.size(yy,1)), dtype=complex)
    I01 = np.zeros((np.size(xx,0),np.size(yy,1)), dtype=complex)
    I02 = np.zeros((np.size(xx,0),np.size(yy,1)), dtype=complex)
    Ex = np.zeros((np.size(xx,0),np.size(yy,1)), dtype=complex)
    Ey = np.zeros((np.size(xx,0),np.size(yy,1)), dtype=complex)
    Ez = np.zeros((np.size(xx,0),np.size(yy,1)), dtype=complex)
    
    E_const = (1j*k*f/2.0) * np.sqrt(1/n) * E0 * np.exp(-1j*k*f)

    for i in range(0,np.size(xx,0)):
        for j in range(0, np.size(yy,1)):
            I00[i,j] = complex_int(I00_integrand, 0, t_max, args=(f_w, t_max, k, rho[i,j], 0, w0, f))
            I01[i,j] = complex_int(I01_integrand, 0, t_max, args=(f_w, t_max, k, rho[i,j], 0, w0, f))
            I02[i,j] = complex_int(I02_integrand, 0, t_max, args=(f_w, t_max, k, rho[i,j], 0, w0, f))
            Ex[i,j] = E_const * (I00[i,j]+I02[i,j]*np.cos(2*phi[i,j]))
            Ey[i,j] = E_const * (I02[i,j]*np.sin(2*phi[i,j]))
            Ez[i,j] = E_const * (-2j*I01[i,j]*np.cos(phi[i,j]))

    return (Ex, Ey, Ez, xx, yy)
