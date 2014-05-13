from focused_beam import *
import matplotlib.pyplot as plt

def main():
    
    # Common parameters
    wl = 800e-9 # Wavelength in m
    NA = 1.4 # lens numerical aperture
    n = 1.518 # refractive index of the media
    f = 500e-6 # focal length in m

    w0 = 500e-6 # Beam waist at back aperture of the lens (overfills objective)

    t_max = np.arcsin(NA/n) # theta_max in radians
    k = 2.0*np.pi*n/wl # wavenumber in the medium
    
    E0 = 1 
    
    # Y, Z plane fields, Phi = pi/2
    #y = np.linspace(-2*wl, 2*wl, 160)
    #z = np.linspace(-2*wl, 2*wl, 160)
    #(Ex, Ey, Ez, yy, zz) = compute_YZ_fields_TEM00(y, z, t_max, k, E0, f, n, w0)
    
    #Ex_msq = np.abs(Ex)**2
    #Ey_msq = np.abs(Ey)**2
    #Ez_msq = np.abs(Ez)**2
    #E_msq = Ex_msq + Ey_msq + Ez_msq
    
    #ywl = yy/wl; zwl = zz/wl; # coordinates normalized by wavelength
    #plt.pcolormesh(ywl, zwl, E_msq)
    #plt.colorbar()
    #plt.title('|E|^2')

    # X, Y plane at focus
    x = np.linspace(-2*wl, 2*wl, 60)
    y = np.linspace(-2*wl, 2*wl, 60)

    # Circularly polarized beam: Superposed x polarized and y polarized beams 
    # with a 90 degree phase shift. 
    (Ex1, Ey1, Ez1, xx, yy) = compute_XY_fields_TEM00(x, y, t_max, k, E0, f, n, w0, pol='y')
    (Ex2, Ey2, Ez2, xx, yy) = compute_XY_fields_TEM00(x, y, t_max, k, 1j*E0, f, n, w0, pol='x')

    Ex_msq = np.abs(Ex1+Ex2)**2
    Ey_msq = np.abs(Ey1+Ey2)**2
    Ez_msq = np.abs(Ez1+Ez2)**2
    E_msq = Ex_msq + Ey_msq + Ez_msq

    xwl = xx/wl; ywl = yy/wl; # coordinates normalized by wavelength
    fig = plt.figure()
    fig.suptitle('X-Y plane')
    plt.subplot(221)
    plt.pcolormesh(xwl, ywl, Ex_msq)
    plt.colorbar()
    plt.title('|Ex|^2')
    plt.subplot(222)
    plt.pcolormesh(xwl, ywl, Ey_msq)
    plt.colorbar()
    plt.title('|Ey|^2')
    plt.subplot(223)
    plt.pcolormesh(xwl, ywl, Ez_msq)
    plt.colorbar()
    plt.title('|Ez|^2')
    plt.subplot(224)
    plt.title('|Ez|^2')
    plt.pcolormesh(xwl, ywl, E_msq)
    plt.colorbar()
    plt.title('|E|^2')
    plt.show()

if __name__ == "__main__":
    main()
