import morphine
import numpy

import jax
import jax.numpy as np
from jax import jit, vmap, grad

# ---------
# Optics functions
# ---------

def lsq_params(img):
    xx, yy = np.meshgrid(np.linspace(0,1,img.shape[0]),np.linspace(0,1,img.shape[1]))
    A = np.vstack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]).T
    matrix = np.linalg.inv(np.dot(A.T,A)).dot(A.T)
    return matrix, xx, yy, A

@jit
def lsq(img):
    matrix, _, _, _ = lsq_params(img)
    return np.dot(matrix,img.ravel())

@jit
def jit_area(img, epsilon = 1e-15):

    a,b,c = lsq(img)
    a, b, c = np.where(a==0,epsilon,a), np.where(b==0,epsilon,b), np.where(c==0,epsilon,c)
    x1 = (-b-c)/(a) # don't divide by zero
    x2 = -c/(a) # don't divide by zero
    x1, x2 = np.min(np.array([x1,x2])), np.max(np.array([x1,x2]))
    x1, x2 = np.max(np.array([x1,0])), np.min(np.array([x2,1]))

    dummy = x1 + (-c/b)*x2-(0.5*a/b)*x2**2 - (-c/b)*x1+(0.5*a/b)*x1**2

    # Set the regions where there is a defined gradient
    dummy = np.where(dummy>=0.5,dummy,1-dummy)
    
    # Colour in regions
    dummy = np.where(np.mean(img)>=0,dummy,1-dummy)
    
    # rescale between 0 and 1?
    dummy = np.where(np.all(img>0),1,dummy)
    dummy = np.where(np.all(img<=0),0,dummy)

    # undecided region
    dummy = np.where(np.any(img==0),np.mean(dummy>0),dummy)

    # rescale between 0 and 1
    dummy = np.clip(dummy, 0, 1)
    
    return dummy


@jit
def soft_binarise_wf(wf, ppsz = 256):
    psz = ppsz * 3

    dummy = np.array(wf.split(ppsz))
    dummy = np.array(dummy.split(ppsz, axis = 2))
    subarray = dummy[:,:,0,0]

    flat = dummy.reshape(-1, 3, 3)
    vmap_mask = vmap(jit_area, in_axes=(0))

    soft_bin = vmap_mask(flat).reshape(ppsz, ppsz)
    
    return soft_bin

@jit
def get_psf(binary, support, pdiam = 1, isz = 128, pixscale = 0.025, wavelength = 1.0e-6, ppsz = 256):
    '''psf from soft binarised zernikes'''
    
    ppscale = pdiam/ppsz
    psz = ppsz * 3

    binary_pupil = morphine.ArrayOpticalElement(opd=binary*wavelength/2.,
                                                transmission=np.abs(support.astype(float)),
                                                pixelscale=ppscale,
                                                name='mask',planetype=1)
    
    osys = morphine.OpticalSystem(npix = ppsz)
    osys.add_pupil(binary_pupil)
    osys.pupil_diameter = pdiam
    osys.add_detector(pixelscale = pixscale * 2, fov_arcsec = 2.*isz*pixscale)
    
    blur, _ = osys.propagate_mono(wavelength, normalize = 'first')
    
    wf = binary.reshape(ppsz, ppsz)
    psf = blur.intensity
    
    return psf, wf

# ---------
# Gradient energy
# ---------

@jit
def spatial_grad(im):
    
    # Get pad values
    npix, ypad, xpad = im.shape[0], 2*im.shape[0], 2*im.shape[1]
    paddingsx = [[0, ypad-1], [0, xpad-3]]
    paddingsy = [[0, ypad-3], [0, xpad-1]]

    # Create convolution vectors
    xvec = np.array([[-1, 0, 1]])
    yvec = np.array([[-1], [0], [1]])

    # Pad vectors
    pad_im = np.pad(im, [[0, npix], [0, npix]])
    padx = np.pad(xvec, paddingsx)
    pady = np.pad(yvec, paddingsy)

    # Create conplex arrays for transforming
    pad_im_cplx = np.complex128(pad_im)
    padx_cplx = np.complex128(padx)
    pady_cplx = np.complex128(pady)

    # Transform
    ft_im = np.fft.fft2(pad_im_cplx)
    ftx = np.fft.fft2(padx_cplx)
    fty = np.fft.fft2(pady_cplx)

    # Multiply
    multx = ft_im * ftx
    multy = ft_im * fty

    # Obtain gradients
    xout_full = np.real(np.fft.ifft2(multx))/2
    yout_full = np.real(np.fft.ifft2(multy))/2

    # De-pad
    xout = xout_full[:npix, :npix]
    yout = yout_full[:npix, :npix]
    
    return xout, yout

@jit
def calc_ftrwge(im, rmax = 100):
    Y, X = spatial_grad(im)

    # Generate R and theta
    size = im.shape[0]
    r = size/2
    xx, yy = np.meshgrid(np.linspace(-r, r-1, size), np.linspace(-r, r-1, size))
    rr = np.sqrt(xx**2 + yy**2)
    tth = np.angle(xx+1.j*yy)

    # Get radial scaling value and multiply x and y values by these
    xcomp = X * rr * np.cos(tth)
    ycomp = Y * rr * np.sin(tth)

    # Add and square
    val = np.square(xcomp + ycomp)

    mask = rr < rmax
    masked = mask * val
    
    out = np.sum(masked)
    
    return out

def calc_ge(im, rmax = 100):
    Y, X = spatial_grad(im)
    out = np.sum(X**2 + Y**2)
    
    return out

# ---------
# Helper
# ---------

def radius_meshgrid(N):
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    return np.sqrt(xx**2 + yy**2)