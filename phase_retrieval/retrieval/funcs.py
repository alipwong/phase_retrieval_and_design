from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


def pad(x, factor = 2, center = False):
    """
    Takes in an image and pads it with 0 so that the size increases by the specified factor. If center = True, it puts the original image in the center. Note: This is inaccurate if the img has odd dimensions.
    """
    pad = int(x.shape[0]*(factor-1))
    if center:
        if pad % 2 != 0:
            print("WARNING! pad size odd. Image size may not be accurate")
        pad = pad // 2
        return tf.pad(x, [[pad, pad,], [pad, pad]], "CONSTANT")
    return tf.pad(x, [[0, pad,], [0, pad]], "CONSTANT")


def FFT(x):
    return tf.signal.fftshift(tf.signal.fft2d(x))


def intensity(x):
    """Takes an image and returns the intensity (absolute value squared).
    """
    return tf.math.square(tf.math.abs(x))

def build_pupil(phases, mag, gridsize = 128):


    re = tf.multiply(mag, tf.math.cos(phases)) 
    im = tf.multiply(mag, tf.math.sin(phases)) 

    pupil = tf.complex(re, im)
    return pupil


def normalise(img):
    return img/tf.reduce_sum(img)

def prop_psf(pupil, factor):

    pad_pupil = pad(pupil, factor = factor)
    return intensity(FFT(pad_pupil))

###
# Convenience plotting
###

def show(phases, psf, history, start, epochs):
    
    plt.figure(figsize = (20, 3))
    plt.subplot(151)
    plt.imshow(phases%(2*np.pi), interpolation = 'none', cmap = 'twilight', vmin = 0, vmax = 2*np.pi)
    plt.title("Pupil phases")
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(152)
    plt.imshow(psf, interpolation = 'none')
    plt.title("PSF")
    plt.axis('off')
    plt.colorbar()
    
    plt.subplot(153)
    c = psf.shape[0]//2
    s = 50
    plt.imshow(psf[c-s:c+s, c-s:c+s], interpolation = 'none')
    plt.title("PSF (zoomed)")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(154)
    plt.plot(range(start, epochs + start), history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel("Loss")

    plt.subplot(155)
    plt.plot(range(start, epochs + start), history.history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel("Learning Rate")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.yscale("log")
    
    plt.subplots_adjust(wspace = 0.4)

    plt.show()
    
def show_binarisation(phases, history, start, epochs):
    
    plt.figure(figsize = (8, 3))

    plt.subplot(121)
    plt.hist(phases.numpy().flatten()%(2*np.pi), bins = 180)
    plt.title("Histogram of phases")
    
    ax = plt.subplot(122)
    ax.plot(range(start, epochs + start), history.history['error_loss'], label = 'Error', color = 'tab:blue')
    ax.set_ylabel('error', color = 'tab:blue')
    ax2 = ax.twinx()
    ax2.plot(range(start, epochs + start), history.history['binarise_loss'], label = 'Binarise', color = 'tab:orange')
    ax2.set_ylabel('Binarise', color = 'tab:orange')
    plt.xlabel('Epoch')
    
    plt.subplots_adjust(wspace = 0.4)

    plt.show()
    
# def show_binarisation(phases, history, start, epochs):
    
#     plt.figure(figsize = (8, 3))

#     plt.subplot(121)
#     plt.hist(phases.numpy().flatten()%(2*np.pi), bins = 180)
#     plt.title("Histogram of phases")
    
#     plt.subplot(122)
#     plt.plot(range(start, epochs + start), history.history['error_loss'], label = 'Error')
#     plt.plot(range(start, epochs + start), history.history['binarise_loss'], label = 'Binarise')
#     plt.xlabel('Epoch')
#     plt.legend()
    
#     plt.subplots_adjust(wspace = 0.4)

#     plt.show()