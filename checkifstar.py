#!/usr/bin/env python

import numpy as np
import os
from astropy.io import fits
from jwst.datamodels import dqflags
import photutils
from photutils import CircularAnnulus
from datetime import datetime
import webbpsf
from copy import deepcopy

def checkif(cutout,cutoutdq,webbpsfcutoutmask,radmin,radmax,spikeratio):
    """
    Given a NIRISS 2D image cutout check for diffraction spikes and return True or False for it being a star (or return a probability?)
    Parameters:
    cutout - 2D image cutout
    cutoutdq - 2D image cutout dq array
    webbpsfcutoutmask - same size and centered cutout from WebbPSF for this filter as a mask with values of 2 in spikes and 1 off spikes (0 elsewhere)
    radmin - starting radius in pixels  to look for diffraction spikes
    radmax - ending radius in pixels to look for diffraction spikes
    spikeratio - ratio of mean flux in spike region to out of spike region above which to call it a star for one annulus 
    Returns:
    isstar - True or False
    """

    #Edit webbpsfcutoutmask to mask bad pixels
    webbpsfcutoutmaskhere = deepcopy(webbpsfcutoutmask)
    donotuseindices = np.where(np.bitwise_and(cutoutdq, dqflags.pixel['DO_NOT_USE']))
    webbpsfcutoutmaskhere[donotuseindices] = 0

    #Subtract background off - use 5%-ile of non-saturated array pixels
    nonsatmask = np.ones((cutout.shape))
    saturatedindices = np.where(np.bitwise_and(cutoutdq, dqflags.pixel['SATURATED']))
    nonsatmask[saturatedindices] = 0
    cutoutnonsat = cutout[np.where(nonsatmask>0)]
    #Only continue if cutout is same shape as webbpsf mask 
    if cutout.shape==webbpsfcutoutmaskhere.shape: 
        fifthpercentile = np.percentile(cutoutnonsat,5)
        cutout -= fifthpercentile

        #Set up separate masks for each region
        maskspikes = webbpsfcutoutmaskhere-1
        maskspikes[maskspikes<0] = 0
        masknotspikes = webbpsfcutoutmaskhere
        masknotspikes[np.where(webbpsfcutoutmaskhere>1)] = 0

        #Loop over annuli
        psfcenxy = (cutout.shape[0]-1)/2
        numrad = radmax-radmin+1
        radii = np.arange(numrad)+radmin
        yesspike=0
        
        for k in range(numrad):
            annulus_aperture = CircularAnnulus([psfcenxy,psfcenxy], r_in=radii[k]-0.5, r_out=radii[k]+0.5)

            #First for spikes
            annulus_mask = annulus_aperture.to_mask(method='center')
            annulus_measuredata = annulus_mask.multiply(cutout*maskspikes)
            annulus_maskdata = annulus_mask.data
            annulus_measuredata_1d = annulus_measuredata[annulus_maskdata > 0]
            spikes_mean = np.mean(annulus_measuredata_1d)

            #Then out of spikes
            annulus_mask = annulus_aperture.to_mask(method='center')
            annulus_measuredata = annulus_mask.multiply(cutout*masknotspikes)
            annulus_maskdata = annulus_mask.data
            annulus_measuredata_1d = annulus_measuredata[annulus_maskdata > 0]
            notspikes_mean = np.mean(annulus_measuredata_1d)

            #Keep count of annuli where spikes are brighter
            if spikes_mean/notspikes_mean > spikeratio:
                yesspike+=1

        fracyesspike = yesspike/numrad
        if fracyesspike >= 0.5:
            isstar = True
        else:    
            isstar = False
        print (fracyesspike,isstar)
    else:
        isstar = False
    return isstar


def makewebbpsfmask(ins,filtername,pixscale,cutsize,radmin,radmax):
    """
    make  WebbPSF cutout mask for this filter with values of 1 in spikes and 2 off spikes (0 elsewhere)
    Requires WEBBPSF_PATH environment variable to have been set.
    Parameters:
    ins - instrument NIRISS or NIRCAM
    filtername - filter name as understood by webbpsf
    pixscale - pixel scale in arcsec per pixel
    cutsize - 2D image cutout length along one axis in pixels - should be odd to center PSF -  will be a square - start with 37 pixels for NIRISS
    Returns:
    webbpsfcutoutmask - same size and centered cutout from WebbPSF for this filter as a mask with values of 2 in spikes and 1 off spikes (0 elsewhere)
    """

    outfile='temppsfmask.fits'
    psffile="./WebbPSF_{}_{}_{}_{}.fits".format(ins,filtername,str(pixscale).replace('.','p'),cutsize)
    if not os.path.exists(psffile):
        if ins.lower()=='niriss':
            nisornic = webbpsf.NIRISS()
        else:
            nisornic = webbpsf.NIRCam()
        nisornic.options['output_mode'] = 'detector sampled'
        nisornic.pixelscale = pixscale
        nisornic.filter=filtername
        psfhdulist = nisornic.calc_psf(psffile,fov_pixels=cutsize)
    else:
        psfhdulist = fits.open(psffile)
    psf=psfhdulist[0].data
    psf /= psf.sum()

    #For NIRISS F090W to F200W start at 9 pixels radius and go out to 16 pixels radius
    webbpsfcutoutmask = np.zeros((cutsize,cutsize))
    psfcenxy = (cutsize-1)/2
    numrad = radmax-radmin+1
    radii = np.arange(numrad)+radmin
    for k in range(numrad):
        annulus_aperture = CircularAnnulus([psfcenxy,psfcenxy], r_in=radii[k]-0.5, r_out=radii[k]+0.5)
        annulus_mask = annulus_aperture.to_mask(method='center')
        masksize = annulus_mask.shape[0]
        xlo = int((cutsize-masksize)/2)
        xhi = xlo+masksize
        ylo=xlo
        yhi=xhi

        annulus_measuredata = annulus_mask.multiply(psf)
        annulus_maskdata = annulus_mask.data

        annulus_measuredata_1d = annulus_measuredata[annulus_maskdata > 0]
        annulus_measuredata_median=np.median(annulus_measuredata_1d)
        annulus_measuredata_std=np.std(annulus_measuredata_1d)
        thresh = annulus_measuredata_median+0.0*annulus_measuredata_std
        annulus_maskdata[np.where(annulus_measuredata>thresh)] = 2
        webbpsfcutoutmask[ylo:yhi,xlo:xhi] = webbpsfcutoutmask[ylo:yhi,xlo:xhi] + annulus_maskdata
    psfhdulist[0].data = webbpsfcutoutmask
    psfhdulist.writeto(outfile,overwrite=True)    
    return webbpsfcutoutmask


#Run directly for testing
direct=False
if direct:
    ins='niriss'
    filtername='F115W'        
    pixscale=0.0656
    cutsize=37
    radmin = 9
    radmax = 16
    spikeratio = 1.4

    #Make the WebbPSF mask
    webbpsfcutoutmask = makewebbpsfmask(ins,filtername,pixscale,cutsize,radmin,radmax)
   
    #Feed the cutouts and see if matches a star
    ratefile='jw01324001001_02101_00001_nis_rate.fits'
    with fits.open(ratefile) as hdulist:
        cutout = hdulist['SCI'].data[1283:1320,1310:1347]
        cutoutdq = hdulist['DQ'].data[1283:1320,1310:1347]
        print (cutout.shape)
    isstar = checkif(cutout,cutoutdq,webbpsfcutoutmask,radmin,radmax,spikeratio)
    print (isstar)
