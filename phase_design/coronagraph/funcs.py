import jax.numpy as np
import morphine

# ---------
# Optics functions
#----------

def get_psf(wf, support, pdiam = 8, isz = 128, pixscale = 0.005, wavelength = 1.6e-6, psz = 256):
    '''psf from soft binarised zernikes'''
    
    ppscale = pdiam/psz
    
    osys = morphine.OpticalSystem(npix = psz)
    osys.pupil_diameter = pdiam

    binary_pupil = morphine.ArrayOpticalElement(opd=wf*wavelength/2.,
                                                transmission = np.abs(support.astype(float)),
                                                pixelscale=ppscale,
                                                name='mask',planetype=1)
    
    osys.add_pupil(binary_pupil)
    osys.add_detector(pixelscale = pixscale * 2, fov_arcsec = 2.*isz*pixscale)
    
    blur, _ = osys.propagate_mono(wavelength, normalize = 'first')
    
    psf = blur.intensity
    
    return psf, wf

def get_support(PSZ, secondary_ratio, spider_width):

    xx, yy = np.meshgrid(np.linspace(-1, 1, PSZ), np.linspace(-1, 1, PSZ))
    rr = np.sqrt(xx**2 + yy**2)
    support = (rr < 1) * (rr > secondary_ratio)

    spy1 = ((1 * xx + yy > -spider_width) * (1 * xx + yy < spider_width)).astype(float)
    spy2 = ((-1 * xx + yy > -spider_width) * (-1 * xx + yy < spider_width)).astype(float)

    spiders = (spy1 + spy2) * support

    support = support & np.logical_not(spiders)
    
    return support

# ---------
# Helper functions
#----------

def radius_meshgrid(N):
    xx, yy = np.meshgrid(np.linspace(-1, 1, N*2), np.linspace(-1, 1, N*2))
    return np.sqrt(xx**2 + yy**2)