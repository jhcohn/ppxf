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

import matplotlib.pyplot as plt
import numpy.random


def ppxf_nifs_kinematics(newgal=None, resid=None, centers=None, coords=[0, 0]):

    file_dir = path.dirname(path.realpath(__file__))  # path of this procedure

    # Read a galaxy spectrum and define the wavelength range
    file = file_dir + '/spectra/pgc12557_combined.fits'  # '/spectra/NGC4550_SAURON.fits'  # my current file location
    hdu = fits.open(file)
    gal_lin = hdu[1].data  # gal_lin = hdu[0].data
    h1 = hdu[1].header  # I need to use 1st extension header (0=general, 1=science, 2=variance, 3=data quality flags)

    lamRange1 = h1['CRVAL3'] + np.array([0., h1['CD3_3']*(h1['NAXIS3'] - 1)])  # [ 19971.86914062  24319.31070422]
    print(lamRange1, 'l1')

    # print(gal_lin[0][20]) all 0s  # print((gal_lin[300][35])) NOT all 0s!
    # len(gal_lin) = 2040, len(gal_lin[0]) = 69, len(gal_lin[0][1]) = 71
    # 2040 --> NAXIS 3, 69 --> NAXIS2, 71 --> NAXIS1
    # There's a len 2040 spectrum at each gal_lin[:,x,y] --> gal_lin[:, x, y] is an array len(2040) starting at
    # lamRange1[0] and finishing at lamRange1[1], with each pixel in between separated by h1['CD3_3'].

    # CUT SPECTRUM TO WAVELENGTH RANGE 2.26 - 2.42
    low_lim = 22600.
    up_lim = 24200.
    cut1 = int((low_lim - lamRange1[0]) / h1['CD3_3'])  # num pixels between pix 1 & pix corresponding to 2.26 microns
    cut2 = int((up_lim - lamRange1[0]) / h1['CD3_3'])  # num pixels between pix 1 & pix corresponding to 2.42 microns
    # print(cut1, cut2, 'cuts')  # 1232.62354281, 1983.04191009 --> int(cut1) = 1232, int(cut2) = 1983
    gal_lin = gal_lin[cut1:cut2]  # cut gal_lin spectrum to new wavelength range
    start = h1['CRVAL3'] + h1['CD3_3'] * cut1
    stop = h1['CRVAL3'] + h1['CD3_3'] * cut2
    lamRange1 = [start, stop]  # redefine lamRange1 to correspond to new wavelength range
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
    x = coords[0]
    y = coords[1]
    # if newgal is None:
    galaxy, logLam1, velscale = util.log_rebin(lamRange1, gal_lin[:, x, y])  # no input velscale --> fcn returns it
    print(len(galaxy), 'len gal')  # 750 now because of cut to gal_lin!
    print(np.median(galaxy))
    galaxy = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    # else:
    #     galaxy, logLam1, velscale = util.log_rebin(lamRange1, newgal)  # newgal is the spectrum at coords x, y

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
    print('shape', noise.shape)  # 751,
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

    lamRange2 = h2['CRVAL1'] + np.array([0., h2['CD1_1']*(h2['NAXIS1'] - 1)])  # BUCKET want NAXIS - 1?
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
    if newgal is None:
        pp = ppxf(templates, galaxy, noise, velscale, start,
                  goodpixels=goodPixels, plot=True, moments=4,
                  degree=4, vsyst=dv, velscale_ratio=1, bias=0.)  # velscale_ratio=velscale_ratio)

        stuff = pp.bestfit, pp.galaxy, pp.sol, goodPixels
    else:
        pp = ppxf(templates, galaxy, noise, velscale, start,
                  goodpixels=goodPixels, plot=True, moments=4,
                  degree=4, vsyst=dv, velscale_ratio=1, bias=0.)  # velscale_ratio=velscale_ratio)
        pp_new = ppxf(templates, newgal, noise, velscale, start,
                      goodpixels=goodPixels, plot=False, moments=4,
                      degree=4, vsyst=dv, velscale_ratio=1, bias=0.)  # velscale_ratio=velscale_ratio)
        #   pp.plot()      # Plot best fit and gas lines
        x_ax = np.arange(galaxy.size)
        plt.plot(x_ax, pp.galaxy, 'k')
        plt.plot(x_ax, newgal, 'b')

        stuff = pp_new.bestfit, pp_new.galaxy, pp_new.sol

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

    return stuff  # output_spectrum, output_noise  # for use in noise_in

# ------------------------------------------------------------------------------


if __name__ == '__main__':

    # NEW ISSUE: why is everything being shifted to the left???
    coords = [33, 35]
    fig = plt.figure()
    out = ppxf_nifs_kinematics(coords=coords)
    sols = out[2]  # pp.sol

    textx = 500
    texty = 1.2
    sols = out[2]
    textx = [375, 475, 575, 675]
    texty = 1.2
    texty2 = 1.15
    fs = 20
    plt.text(textx[0], texty, r'V', size=fs)
    plt.text(textx[1], texty, r'$\sigma$', size=fs)
    plt.text(textx[2], texty, r'h$_3$', size=fs)
    plt.text(textx[3], texty, r'h$_4$', size=fs)
    plt.text(textx[0], texty2, str(int(sols[0])), size=fs)
    plt.text(textx[1], texty2, str(int(sols[1])), size=fs)
    plt.text(textx[2], texty2, str(float('%.2g' % sols[2])), size=fs)
    plt.text(textx[3], texty2, str(float('%.2g' % sols[3])), size=fs)
    fold = '/Users/jonathancohn/Documents/ppxf/figs/'
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(fold + 'ppxf_iteration_0', dpi=500)
    plt.close()
    # plt.show()

    iters = 300
    sol_matrix = np.zeros(shape=(4, iters))
    resid = out[1] - out[0]  # galaxy - bestfit = resid, from ppxf()
    resid = resid[out[3]]  # resid[goodPixels]
    std_dev = np.std(resid)
    bfit = out[0]
    orig = out[2]

    n_above = 0
    n_below = 0
    counter = 0
    for i in range(iters):
        print(counter)
        new_gal = np.full_like(bfit, 0.0)
        for l in range(len(bfit)):
            new_gal[l] += np.random.normal(loc=bfit[l], scale=std_dev, size=1)
            if new_gal[l] - bfit[l] < 0:
                n_below += 1
            else:
                n_above += 1
        print(n_below, 'nb')
        print(n_above, 'na')

        counter += 1
        new_gal = np.asarray(new_gal)
        print(new_gal.shape)

        fig = plt.figure()
        out = ppxf_nifs_kinematics(newgal=new_gal, coords=coords)
        sols = out[2]

        plt.text(textx[0], texty, r'V', size=fs)
        plt.text(textx[1], texty, r'$\sigma$', size=fs)
        plt.text(textx[2], texty, r'h$_3$', size=fs)
        plt.text(textx[3], texty, r'h$_4$', size=fs)
        plt.text(textx[0], texty2, str(int(sols[0])), size=fs)
        plt.text(textx[1], texty2, str(int(sols[1])), size=fs)
        plt.text(textx[2], texty2, str(float('%.2g' % sols[2])), size=fs)
        plt.text(textx[3], texty2, str(float('%.2g' % sols[3])), size=fs)
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(fold + 'ppxf_iteration_' + str(counter))  # , dpi=500)
        sol_matrix[:, counter - 1] = sols  # save solutions
        plt.close()
        #plt.clf()
        #plt.cla()

    print(sol_matrix)
    fig = plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    axes = [ax1, ax2, ax3, ax4]
    labels = [r'V', r'$\sigma$', r'h$_3$', r'h$_4$']
    for i in range(len(sol_matrix)):  # 4
        percs = np.percentile(sol_matrix[i], [0.3, 2.4, 16.0, 50.0, 84.0, 97.6, 99.7])  # median, 1,2,3 sigma
        t_x = min(sol_matrix[i])
        t_y1 = 0.02
        t_y2 = 0.03
        t_y3 = 0.04
        # axes[i].text(t_x, t_y3,)

        axes[i].axvline(x=orig[i], color='k', linestyle='--', lw=2, label='Original fit')

        axes[i].hist(sol_matrix[i], bins=50, histtype="step", weights=[1. / iters] * len(sol_matrix[i]), normed=False,
                     color='k', lw=2, label=labels[i])
        axes[i].axvline(x=percs[3], color='b', linestyle='--', lw=2, label='Median')
        axes[i].axvline(x=np.mean(sol_matrix[i]), color='b', ls='-', lw=2, label='Mean')

        axes[i].axvspan(percs[0], percs[6], color='b', alpha=0.2)  # 3 sigma
        axes[i].axvspan(percs[1], percs[5], color='b', alpha=0.2)  # 2 sigma
        axes[i].axvspan(percs[2], percs[4], color='b', alpha=0.2)  # 1 sigma
        axes[i].legend(numpoints=1, loc='upper left', prop={'size': 10})

        fig.set_size_inches(18.5, 10.5)
        # plt.text()
        plt.savefig(fold + 'ppxf_hists_' + str(iters) + '_iterations')  # , dpi=500)
        print(percs)
        print(orig, 'orig')
    plt.show()
    # print(len(sols))  # 201
    # print(len(sols[0]))  # 4
    # print(len(sols[:]))  # 201

    # plt.pause(1)
