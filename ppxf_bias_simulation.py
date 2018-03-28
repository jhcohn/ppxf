#!/usr/bin/env python
#----------------------------------------------------------------------------
#
# Usage example for the procedure PPXF, which implements the
# Penalized Pixel-Fitting (pPXF) method originally described in
# Cappellari M., & Emsellem E., 2004, PASP, 116, 138
#     http://adsabs.harvard.edu/abs/2004PASP..116..138C
# and upgraded in Cappellari M., 2017, MNRAS, 466, 798
#     http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
#
# This example it is useful to determine the desired value for
# the BIAS keyword of the pPXF procedure. This procedure generates
# a plot similar (but not identical) to Figure 6 in
# Cappellari & Emsellem (2004).
#
# A rough guideline to determine the BIAS value is the following: choose the *largest*
# value which make sure that in the range sigma>3*velScale and for (S/N)>30 the true values
# for the Gauss-Hermite parameters are well within the rms scatter of the measured values.
# See the documentation in the file ppxf.pro for a more accurate description.
#
# MODIFICATION HISTORY:
#   V1.0.0: By Michele Cappellari, Leiden, 28 March 2003
#   V1.1.0: Included in the standard PPXF distribution. After feedback
#       from Alejandro Garcia Bedregal. MC, Leiden, 13 April 2005
#   V1.1.1: Adjust GOODPIXELS according to the size of the convolution kernel.
#       MC, Oxford, 13 April 2010
#   V1.1.2: Use Coyote Graphics (http://www.idlcoyote.com/) by David W. Fanning.
#       The required routines are now included in NASA IDL Astronomy Library.
#       MC, Oxford, 29 July 2011
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 9 December 2013
#   V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V2.0.2: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#   V2.0.3: Use random input velocity to properly simulate situations with
#       undersampled LOSVD. MC, Oxford, 20 April 2016
#   V2.1.0: Replaced the Vazdekis-99 SSP models with the Vazdekis+10 ones.
#       Modified plot to emphasize undersampling effect. MC, Oxford, 3 May 2016
#   V2.1.1: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
#   V2.1.2: Updated MILES file name. MC, Oxford, 29 November 2017
#
##############################################################################

from __future__ import print_function

from time import clock
from os import path

from astropy.io import fits
from scipy import ndimage, signal
import numpy as np
import matplotlib.pyplot as plt

from ppxf import ppxf
import ppxf_util as util

#----------------------------------------------------------------------------

def rebin(x, factor):
    """
    Rebin a one-dimensional vector by averaging
    in groups of "factor" adjacent values

    """
    return np.mean(x.reshape(-1, factor), axis=1)

#----------------------------------------------------------------------------

def ppxf_example_simulation():

    file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

    hdu = fits.open(file_dir + '/veltemps/bd-013097_M2III_rebinflux_rest.fits')  # BUCKET: which star should I choose???
    # '/miles_models/Mun1.30Zp0.00T12.5893_iPp0.00_baseFe_linear_FWHM_2.51.fits')  # Solar metallicitly, Age=12.59 Gyr
    ssp = hdu[1].data  # hdu[1] is science header for us; hdu[0] is generic header
    h = hdu[1].header

    lamRange = h['CRVAL1'] + np.array([0.,h['CD1_1']*(h['NAXIS1']-1)])
    c = 299792.458 # speed of light in km/s
    velscale = c*h['CD1_1']/max(lamRange)   # Do not degrade original velocity sampling
    star, logLam, velscale = util.log_rebin(lamRange, ssp, velscale=velscale)

    # The finite sampling of the observed spectrum is modeled in detail:
    # the galaxy spectrum is obtained by oversampling the actual observed spectrum
    # to a high resolution. This represent the true spectrum, which is later resampled
    # to lower resolution to simulate the observations on the CCD. Similarly, the
    # convolution with a well-sampled LOSVD is done on the high-resolution spectrum,
    # and later resampled to the observed resolution before fitting with PPXF.

    factor = 10                    # Oversampling integer factor for an accurate convolution
    starNew = ndimage.interpolation.zoom(star, factor, order=3) # This is the underlying spectrum, known at high resolution
    star = rebin(starNew, factor)        # Make sure that the observed spectrum is the integral over the pixels

    # find worst SNR, and worst h3, h4 (highest)
    h3 = 0.1       # Adopted G-H parameters of the LOSVD
    h4 = 0.04  # 0.1
    sn = 30.        # Adopted S/N of the Monte Carlo simulation
    m = 300        # Number of realizations of the simulation
    moments = 4
    velV = np.random.rand(m)  # velocity in *pixels* [=V(km/s)/velScale]  # (len m array of random on [0, 1)
    sigmaV = np.linspace(0.5, 4, m) # Range of sigma in *pixels* [=sigma(km/s)/velScale]  # m evenly spaced [0.5,...,4]

    result = np.zeros((m, moments)) # This will store the results
    t = clock()

    for j, (vel, sigma) in enumerate(zip(velV, sigmaV)):

        dx = int(abs(vel) + 5*sigma)   # Sample the Gaussian and GH at least to vel+5*sigma
        x = np.linspace(-dx, dx, 2*dx*factor + 1) # Evaluate the Gaussian using steps of 1/factor pixels.
        w = (x - vel)/sigma
        w2 = w**2
        gauss = np.exp(-0.5*w2)
        gauss /= np.sum(gauss)  # Normalized total(gauss)=1
        h3poly = w*(2.*w2 - 3.)/np.sqrt(3.)           # H3(y)
        h4poly = (w2*(4.*w2 - 12.) + 3.)/np.sqrt(24.) # H4(y)
        losvd = gauss *(1. + h3*h3poly + h4*h4poly)

        galaxy = signal.fftconvolve(starNew, losvd, mode="same") # Convolve the oversampled spectrum
        # FFT convolution: multiplication in the frequency domain corresponds to convolution in the time domain
        galaxy = rebin(galaxy, factor) # Integrate spectrum into original spectral pixels
        noise = galaxy/sn        # 1sigma error spectrum
        galaxy = np.random.normal(galaxy, noise) # Add noise to the galaxy spectrum
        start = np.array([vel + np.random.uniform(-1, 1), sigma*np.random.uniform(0.8, 1.2)])*velscale  # Convert to km/s

        pp = ppxf(star, galaxy, noise, velscale, start,
                  goodpixels=np.arange(dx, galaxy.size - dx),
                  plot=False, moments=moments, bias=0.4)  # 0.2
        result[j,:] = pp.sol

    print('Calculation time: %.2f s' % (clock()-t))

    # large scale kinematics to estimate dark matter fraction within effective radii, and stellar mass-light ratio
    # mass distribution discribed by GME and dark matter dstribution different
    # work with sarah to use JAM; figure out why we ever use schwarzschild model if JAM is faster (what assumptions are
    # we making/what regimes do we have to be in to use JAM?

    plt.clf()
    plt.subplot(221)
    plt.plot(sigmaV*velscale, result[:,0] - velV*velscale, '+k')  # BUCKET: why is V multiplied by velscale?
    plt.axhline(0, color='r')
    plt.axvline(velscale, linestyle='dashed')
    plt.axvline(2*velscale, linestyle='dashed')
    plt.ylim(-20, 20)
    plt.xlabel(r'$\sigma_{\rm in}\ (km\ s^{-1})$')
    plt.ylabel(r'$V - V_{\rm in}\ (km\ s^{-1})$')
    plt.text(2.05*velscale, -15, r'2$\times$velscale')

    plt.subplot(222)
    plt.plot(sigmaV*velscale, result[:,1] - sigmaV*velscale, '+k')  # BUCKET: why is sigma multiplied by velscale?
    plt.axhline(0, color='r')
    plt.axvline(velscale, linestyle='dashed')
    plt.axvline(2*velscale, linestyle='dashed')
    plt.ylim(-20, 20)
    plt.xlabel(r'$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel(r'$\sigma - \sigma_{\rm in}\ (km\ s^{-1})$')
    plt.text(2.05*velscale, -15, r'2$\times$velscale')

    plt.subplot(223)
    plt.plot(sigmaV*velscale, result[:,2], '+k')
    plt.axhline(h3, color='r')
    plt.axhline(0, linestyle='dotted', color='limegreen')
    plt.axvline(velscale, linestyle='dashed')
    plt.axvline(2*velscale, linestyle='dashed')
    plt.ylim(-0.2+h3, 0.2+h3)
    plt.xlabel(r'$\sigma_{\rm in}\ (km\ s^{-1})$')
    plt.ylabel('$h_3$')
    plt.text(2.05*velscale, h3 - 0.15, r'2$\times$velscale')

    plt.subplot(224)
    plt.plot(sigmaV*velscale, result[:,3], '+k')
    plt.axhline(h4, color='r')
    plt.axhline(0, linestyle='dotted', color='limegreen')
    plt.axvline(velscale, linestyle='dashed')
    plt.axvline(2*velscale, linestyle='dashed')
    plt.ylim(-0.2+h4, 0.2+h4)
    plt.xlabel(r'$\sigma_{\rm in}\ (km\ s^{-1})$')
    plt.ylabel('$h_4$')
    plt.text(2.05*velscale, h4 - 0.15, r'2$\times$velscale')

    plt.tight_layout()
    plt.show() # pause(1)

#----------------------------------------------------------------------------

if __name__ == '__main__':

    np.random.seed(123)  # For reprodcible results
    ppxf_example_simulation()
