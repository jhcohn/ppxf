#!/usr/bin/env python
##############################################################################
#
# Usage example for the procedure PPXF, which implements the
# Penalized Pixel-Fitting (pPXF) method originally described in
# Cappellari M., & Emsellem E., 2004, PASP, 116, 138
#     http://adsabs.harvard.edu/abs/2004PASP..116..138C
# and upgraded in Cappellari M., 2017, MNRAS, 466, 798
#     http://adsabs.harvard.edu/abs/2017MNRAS.466..798C
#
# The example also shows how to include a library of templates
# and how to mask gas emission lines if present.
#
# MODIFICATION HISTORY:
#   V1.0.0: Written by Michele Cappellari, Leiden 11 November 2003
#
##############################################################################

from __future__ import print_function

import glob
from os import path
from time import clock

from astropy.io import fits
from scipy import ndimage
import numpy as np

from ppxf import ppxf
import ppxf_util as util

def ppxf_nifs_kinematics():

    file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

    # Read a galaxy spectrum and define the wavelength range
    #
    file = file_dir + '/spectra/pgc12557_combined.fits'  # '/spectra/NGC4550_SAURON.fits'  # my current file location
    hdu = fits.open(file)
    gal_lin = hdu[1].data  # gal_lin = hdu[0].data
    # h1 = hdu[0].header
    h1 = hdu[1].header  # I need to use 1st extension header (0=general, 1=science, 2=variance, 3=data quality flags)
    '''
    print(h1)
    
    BITPIX  =                  -32 / array data type                                
    NAXIS   =                    3 / number of array dimensions                     
    NAXIS1  =                   71                                                  
    NAXIS2  =                   69                                                  
    NAXIS3  =                 2040                                 
    PCOUNT  =                    0 / number of parameters                           
    GCOUNT  =                    1 / number of groups                               
    EXTNAME = 'SCI     '           / Extension name                                 
    EXTVER  =                    1 / Extension version                              
    INHERIT =                    F / Inherits global header                         
    ORIGIN  = 'NOAO-IRAF FITS Image Kernel July 2003' / FITS file originator        
    OBJECT  = 'PGC12557'           / Name of the object observed                    
    DISPAXIS=                    3 / Dispersion axis                                
    PIXSCALE=                 0.05 / Pixel scale in arcsec/pixel                    
    CTYPE1  = 'LINEAR  '           / coordinate type for the dispersion axis        
    CTYPE2  = 'LINEAR  '           / coordinate type for the spatial axis           
    DC-FLAG =                    0 /                                                
    CD1_1   =                 0.05 /                                                
    CD2_2   =                 0.05 /                                                
    DCLOG1  = 'Transform'                                                           
    WCSDIM  =                    3 /                                                
    CRVAL1  =                2.987                                                  
    CRPIX1  =                  68.                                                  
    CRPIX2  =                   2.                                                  
    WAT1_001= 'wtype=linear axtype=xi' /                                            
    WAT2_001= 'wtype=linear axtype=eta' /                                           
    CTYPE3  = 'LINEAR  '           /                                                
    WAT3_001= 'wtype=linear axtype=wave' /                                          
    CRVAL3  =      19971.869140625 /                                                
    CD3_3   =         2.1321439743 /                                                
    AIRMASS =                1.374                                                  
    CRPIX3  =                   1.                                                  
    LTM1_1  =                   1.                                                  
    LTM2_2  =                   1.                                                  
    LTM3_3  =                   1.                                                  
    WAT0_001= 'system=world'
    END                                     
    '''

    '''
    # NOTE:
    Copied from util.log_rebin():
    lamRange: two elements vector containing the central wavelength
        of the first and last pixels in the spectrum, which is assumed
        to have constant wavelength scale! E.g. from the values in the
        standard FITS keywords: LAMRANGE = CRVAL1 + [0, CDELT1*(NAXIS1 - 1)].
        It must be LAMRANGE[0] < LAMRANGE[1].
    '''
    # lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])  # original
    # lamRange1 = h1['CRVAL3'] + np.array([0., h1['CD3_3']*(h1['CRPIX3'])])  # [ 19971.86914062  19974.0012846 ]
    #  if I use CRPIX3, probably don't want CRPIX3 - 1 because CRPIX3 = 1., and need lamRange1[0] < lamRange1[1]
    lamRange1 = h1['CRVAL3'] + np.array([0., h1['CD3_3']*(h1['NAXIS3'] - 1)])  # [ 19971.86914062  24319.31070422]
    print(lamRange1, 'l1')

    # print(gal_lin[0][20]) all 0s
    # print((gal_lin[300][35])) NOT all 0s!
    # print(len(lamRange1))  # len(lamRange1) = 2, # len(gal_lin) = 2040, len(gal_lin[0]) = 69, len(gal_lin[0][1]) = 71
    # 2040 --> NAXIS 3, 69 --> NAXIS2, 71 --> NAXIS1
    # HMM: want gal_lin to be spectrum itself, which should just be 1d array
    # There's a len 2040 spectrum at each gal_lin[:,x,y]

    # CUT SPECTRUM TO WAVELENGTH RANGE 2.26 - 2.42
    # SO: gal_lin is an array len(2040) starting at lamRange1[0] and finishing at lamRange1[1], with each pixel in
    # between separated by h1['CD3_3']. So find [22600 - lamRange1[0]] / CD3_3, and that should give the number of
    # pixels between pixel 1 and the pixel corresponding roughly to 2.26 microns. Then find [24200 - lamRange1[1]] /
    # CD3_3, which should give the number of pixels between pixel 1 and the pixel corresponding to 2.42 microns.
    start = 22600.
    stop = 24200.
    cut1 = (start - lamRange1[0]) / h1['CD3_3']
    cut2 = (stop - lamRange1[0]) / h1['CD3_3']
    # print(cut1, cut2, 'cuts')  # 1232.62354281, 1983.04191009
    gal_lin = gal_lin[int(cut1):int(cut2)]  # [1233:1983]
    # start1 = h1['CRVAL3'] + h1['CD3_3'] * int(cut1)
    # stop1 = h1['CRVAL3'] + h1['CD3_3'] * int(cut2)
    # print(start1, stop1, 'me')
    lamRange1 = [h1['CRVAL3'] + h1['CD3_3'] * int(cut1), h1['CRVAL3'] + h1['CD3_3'] * int(cut2)]
    print(lamRange1, 'l1, cut')
    # len(gal_lin) is now NOT 2040 but 1983 - 1233 = 750

    FWHM_gal = 4.2  # SAURON has an instrumental resolution FWHM of 4.2A.  # BUCKET: do I need this? If so what for?

    # If the galaxy is at significant redshift, one should bring the galaxy
    # spectrum roughly to the rest-frame wavelength, before calling pPXF
    # (See Sec2.4 of Cappellari 2017). In practice there is no
    # need to modify the spectrum in any way, given that a red shift
    # corresponds to a linear shift of the log-rebinned spectrum.
    # One just needs to compute the wavelength range in the rest-frame
    # and adjust the instrumental resolution of the galaxy observations.
    # This is done with the following three commented lines:
    #
    # z = 1.23 # Initial estimate of the galaxy redshift
    # lamRange1 = lamRange1/(1+z) # Compute approximate restframe wavelength range
    # FWHM_gal = FWHM_gal/(1+z)   # Adjust resolution in Angstrom

    # There's a len 2040 spectrum at each gal_lin[:,x,y]
    x = 33
    y = 35
    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin[:, x, y])  # no input velscale --> function returns
    print(len(galaxy), 'len gal')  # 750 now because of cut to gal_lin!
    galaxy = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    # print(galaxy)

    # BUCKET: constant noise/pix good assumption for me? If so what value do I use? TRY our noise! Maybe trust more?!
    # basically bootstrap the noise!
    # Do one fit with flat noise, then save the best fit spectrum and the residuals.
    # Then, iterate ~200 times. For these iterations, set bias = 0.0. Each iteration, for each pixel, use the spectrum
    # value as the center of a gaussian and use the residuals at that pixel value as the width of the gaussian. Draw
    # from the resultant distribution to make the new noise. For each iteration, save the output V, sigma, h3, h4, and
    # print each spectrum so that we can see it evolving (it should look more like a real spectrum, rather than smooth
    # curve without lines)
    noise = np.full_like(galaxy, 0.0047)  # Assume constant noise per pixel here

    # MEANTIME: why is output velocity close to systemic insted of close to 0, if I'm dealing with redshift already?
    # SET bias = 0
    #
    # print(noise.shape)  # 751,
    # print(galaxy.shape)  # 751,

    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis (2010, MNRAS, 404, 1639) http://miles.iac.es/. A subset
    # of the library is included for this example with permission
    vazdekis = glob.glob(file_dir + '/veltemps/*.fits')  # '/miles_models/Mun1.30Z*.fits')  # my new
    # BUCKET: what are FWHM of spectra in the veltemps library??
    FWHM_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
    # BUCKET: what spectral sampling compared to galaxy do we want for templates??

    # velscale_ratio = 2  # 1.23  # 2  # adopts 2x higher spectral sampling for templates than for galaxy
    # PROBLEM!! If velscale_ratio is not integer, we get issue later because we slice a list with velscale_ratio
    # so need to change velscale? But it's set by util.log_rebin originally! Only need this if oversampling the
    # templates, which I'm not doing

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to a velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    print(vazdekis[0], 'template name')
    hdu = fits.open(vazdekis[0])  # do for just one template to determine size needed for array containing all templates
    ssp = hdu[1].data  # was hdu[0], but that's generic header rather than science header
    h2 = hdu[1].header  # was hdu[0]
    '''
    for line in h2:
        print(line, h2[line])
    
    XTENSION IMAGE
    BITPIX -32
    NAXIS 1
    NAXIS1 1721
    PCOUNT 0
    GCOUNT 1
    EXTNAME SCI
    EXTVER 1
    INHERIT False
    ORIGIN NOAO-IRAF FITS Image Kernel July 2003
    OBJECT BD-01 3097
    DATE 2015-02-15T13:32:06
    IRAF-TLM 2015-02-15T13:32:30
    NFPAD 2015-02-14T14:40:41
    GEM-TLM 2015-02-14T14:40:41
    DISPAXIS 1
    PIXSCALE 0.043
    NSCUTSEC [1:2040,1145:1213]
    NSCUTSPC 13
    FIXPIX Feb 14  8:44 Bad pixel file is tmpdq20234_650
    CTYPE1 LINEAR
    CD1_1 2.13
    CTYPE2 LINEAR
    CD2_2 0.04570113
    DCLOG1 Transform
    DC-FLAG 0
    WCSDIM 3
    CRVAL1 20628.29
    CRPIX1 1.0
    CRPIX2 -28.0
    WAXMAP01 1 0 0 29 0 0
    WAT1_001 wtype=linear axtype=wave
    WAT2_001 wtype=linear axtype=eta
    CTYPE3 LINEAR
    WAT3_001 wtype=linear axtype=xi
    CRVAL3 1.751
    CD3_3 0.103
    LTV2 -29.0
    LTM1_1 1.0
    LTM2_2 1.0
    LTM3_3 1.0
    WAT0_001 system=image
    EXPTIME 25.0
    IMCMB001 xatfbrsnN20070508S0187.fits[SCI]
    IMCMB002 xatfbrsnN20070508S0190.fits[SCI]
    IMCMB003 xatfbrsnN20070508S0191.fits[SCI]
    IMCMB004 xatfbrsnN20070508S0194.fits[SCI]
    NCOMBINE 4
    GAINORIG 1.0
    RONORIG 0.0
    GAIN 2.666667
    RDNOISE 0.0
    '''
    # lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])  # original
    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CD1_1']*(h2['NAXIS1'] - 1)])  # BUCKET want NAXIS - 1?
    # lamRange2 = h2['CRVAL1'] + np.array([0., h2['CD1_1']*(h2['CRPIX1'])])  # or use CRPIX1??
    print(lamRange2, 'l2')  # [ 20628.29  24291.89]
    # print((lamRange2[1] - lamRange2[0])/h2['CD1_1'], 'num of steps in lam2')  # 1720.
    # print(len(ssp), 'len ssp')  # 1721
    sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale)  # /velscale_ratio)
    # print(len(sspNew), 'len sspnew')  # 622 hmmmm NEED THIS TO BE >= (len(galaxy)=750)  # FIXED, now 1791
    templates = np.empty((sspNew.size, len(vazdekis)))

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    # sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
    sigma = FWHM_dif/2.355/h2['CD1_1']  # Sigma difference in pixels

    for j, file in enumerate(vazdekis):  # now for each template file; so why do the thing above with just 1??
        hdu = fits.open(file)
        # ssp = hdu[0].data
        ssp = hdu[1].data
        ssp = ndimage.gaussian_filter1d(ssp, sigma)
        # ndimage.gaussian_filter takes input array (ssp) and filters it. Sigma = standard deviation of Gaussian kernel
        # used to filter the array
        # note: discrete convolution is defined (for any 2 arrays a, v): (a*v)[n] == sum m(-inf to inf) of a[m]*v[n-m]
        # where a is the original curve and v is the gaussian
        # print(len(ssp))  # 1721 --> currently by default being resampled to be 116, want it to be resampled to be 71
        sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale)  # /velscale_ratio)
        # print(velscale_temp, 'vt')  # 78.82967746
        # print(velscale, 'v')  # 78.82967746
        # print(len(sspNew))  # need this to be >= len(galaxy)  # now 1791
        templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

    # print(len(templates[0]))  # len(templates)=29, len(templates[0] = 19)
    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below (see above).
    #
    c = 299792.458
    dv = (logLam2[0] - logLam1[0]) * c  # km/s
    '''
    if velscale_ratio > 1:
        dv = (np.mean(logLam2[:velscale_ratio]) - logLam1[0])*c  # km/s
    else:
        dv = (logLam2[0] - logLam1[0])*c  # km/s
    '''

    z = 0.016561  # z = 0.0015  # Initial redshift estimate of the galaxy
    goodPixels = util.determine_goodpixels(logLam1, lamRange2, z)

    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()
    #  print(galaxy.shape[0])  # = len(galaxy) = 750
    # print(goodPixels)

    # print(max(noise[:, x, y]), min(noise[:, x, y]), np.median(noise[:, x, y]), np.mean(noise[:, x, y]))
    # 0.00106575 -2.77079e-05 4.62628e-05 5.89877e-05
    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodPixels, plot=True, moments=4,
              degree=4, vsyst=dv, velscale_ratio=1)  # velscale_ratio=velscale_ratio)

    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

    print('Elapsed time in pPXF: %.2f s' % (clock() - t))

    # If the galaxy is at significant redshift z and the wavelength has been
    # de-redshifted with the three lines "z = 1.23..." near the beginning of
    # this procedure, the best-fitting redshift is now given by the following
    # commented line (equation 2 of Cappellari et al. 2009, ApJ, 704, L34;
    # http://adsabs.harvard.edu/abs/2009ApJ...704L..34C)
    #
    # print('Best-fitting redshift z:', (z + 1)*(1 + pp.sol[0]/c) - 1)
    return pp.bestfit, pp.galaxy, pp.sol  # output_spectrum, output_noise  # for use in noise_in


#------------------------------------------------------------------------------

if __name__ == '__main__':

    out = ppxf_nifs_kinematics()
    import matplotlib.pyplot as plt

    textx = [375, 475, 575, 675]
    texty = 1.2
    texty2 = 1.15
    sols = out[2]
    fs = 20
    plt.text(textx[0], texty, r'V', size=fs)
    plt.text(textx[1], texty, r'$\sigma$', size=fs)
    plt.text(textx[2], texty, r'h$_3$', size=fs)
    plt.text(textx[3], texty, r'h$_4$', size=fs)
    plt.text(textx[0], texty2, str(int(sols[0])), size=fs)
    plt.text(textx[1], texty2, str(int(sols[1])), size=fs)
    plt.text(textx[2], texty2, str(float('%.2g' % sols[2])), size=fs)
    plt.text(textx[3], texty2, str(float('%.2g' % sols[3])), size=fs)

    plt.show()
    # plt.pause(1)
