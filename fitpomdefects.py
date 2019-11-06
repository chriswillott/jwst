#!/usr/bin/env python

#This procedure takes in direct and dispersed NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror, including the coronagraphic spots. The outputs are maps of the features and images of intensity decrease due to the features. Outputs are provided for two filters because of the shift in image position for the F200W filter compared to the others. 
#Usage, e.g.
#fitpomdefects.py --directflat='./slope/NIST74-F115-FL-6003020347_1_496_SE_2016-01-03T02h19m52_slope_norm.fits'   --gr150cflat='./slope/NIST74-F115-C-FL-6003044302_1_496_SE_2016-01-03T04h57m06_slope_norm.fits' --gr150rflat='./slope/NIST74-F115-R-FL-6003061354_1_496_SE_2016-01-03T06h29m13_slope_norm.fits' --outfile='fitpom115W.fits'

import numpy as np
import optparse
import os, os.path
import tempfile
import astropy.io.fits as fits
from astropy.stats import SigmaClip
from copy import deepcopy
from photutils import detect_threshold,detect_sources,Background2D, MedianBackground,source_properties

# Command line options
op = optparse.OptionParser()
op.add_option("--directflat")
op.add_option("--gr150cflat")
op.add_option("--gr150rflat")
op.add_option("--outfile")

o, a = op.parse_args()
if a:
    print (sys.syserr, "unrecognized option: ",a)
    sys.exit(-1)

directflatfile=o.directflat
gr150cflatfile=o.gr150cflat
gr150rflatfile=o.gr150rflat
outfile=o.outfile

#Read in data and normalise flats
hdulist1=fits.open(directflatfile)
directheader=hdulist1[0].header
directactivedata=hdulist1[0].data[4:2044,4:2044]
hdulist2=fits.open(gr150cflatfile)
gr150cheader=hdulist2[0].header
gr150cactivedata=hdulist2[0].data[4:2044,4:2044]
hdulist3=fits.open(gr150rflatfile)
gr150rheader=hdulist3[0].header
gr150ractivedata=hdulist3[0].data[4:2044,4:2044]
imshape=hdulist1[0].data.shape

normalizedirect=np.median(directactivedata)
normalizegr150c=np.median(gr150cactivedata)
normalizegr150r=np.median(gr150ractivedata)

#Subtract direct off grism flats, so POM features in direct appear positive 
gr150cminusdirect=gr150cactivedata/normalizegr150c-directactivedata/normalizedirect
gr150rminusdirect=gr150ractivedata/normalizegr150r-directactivedata/normalizedirect

#Loop over two grisms
for k in range(2):
    if k==0:
        data=gr150cminusdirect
    else:
        data=gr150rminusdirect
        
    #Subtract a background
    #Don't need a mask excluding low and high pixels as sigma clipping will exclude bright and dark areas
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (24, 24),filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data=data-bkg.background

    #Replace all highly negative pixels (these are dispersed POM features) with zeros
    v=np.where(data<-0.02)
    data[v]=0.0

    #Run source extraction twice
    #Do not use a filter as would make single bright pixels appear as sources
    #Do first run to find small things 
    threshold_small = detect_threshold(data, snr=3.0)
    segm_small = detect_sources(data, threshold_small, npixels=4, connectivity=4, filter_kernel=None) 
    #Do second run to find large things with lower significance and merge the two
    threshold_large = detect_threshold(data, snr=1.5)
    segm_large = detect_sources(data, threshold_large, npixels=100, connectivity=8, filter_kernel=None) 

    #Remove some sources from segmentation image
    catsegm_large = source_properties(data, segm_large)
    numsource=np.array(catsegm_large.label).size
    for ct in range(numsource):
        #Remove bad detector region at x=12, y=1536
        if ((catsegm_large.ycentroid[ct].value>1520)&(catsegm_large.ycentroid[ct].value<1555)&(catsegm_large.xcentroid[ct].value<25)):
            #print ('remove',catsegm_large.label[ct])
            segm_large.remove_labels(labels=catsegm_large.label[ct])
 
    #Remove some sources from segmentation image
    catsegm_small = source_properties(data, segm_small)
    numsource=np.array(catsegm_small.label).size
    for ct in range(numsource):
        print (ct,catsegm_small.label[ct],catsegm_small.xcentroid.value[ct],catsegm_small.ycentroid.value[ct],catsegm_small.area.value[ct]) 
 
        #Remove all sources in lower 100 pixel strip
        if (catsegm_small.ycentroid[ct].value<100.0):
            #print ('remove',catsegm_small.label[ct])
            segm_small.remove_labels(labels=catsegm_small.label[ct])
        #Remove small sources (area<6 pixels) except in top part of detector because POM not in focus so can't be real. 
        if ((catsegm_small.ycentroid[ct].value>100.0 and catsegm_small.ycentroid[ct].value<1700.0 and catsegm_small.area[ct].value<6)):
            #print ('remove',catsegm_small.label[ct])
            segm_small.remove_labels(labels=catsegm_small.label[ct])
        #Remove bad detector region at x=12, y=1536
        if ((catsegm_small.ycentroid[ct].value>1520)&(catsegm_small.ycentroid[ct].value<1555)&(catsegm_small.xcentroid[ct].value<25)):
            #print ('remove',catsegm_small.label[ct])
            segm_small.remove_labels(labels=catsegm_small.label[ct])

        #Remove few artifacts using segemtation IDs - note IDs will change if anything else changes
        #F115-FL-sub-F115-C-FL  threshold = detect_threshold(data, snr=3.0 - no artifacts

        #F115-FL-sub-F115-R-FL  threshold = detect_threshold(data, snr=3.0 - 3 artifacts
        #note x and y below exclude ref pixels
        #x=914  y=2000   S=148 
        #x=531  y=2004   S=150
        #x=856  y=2019   S=152 
        if k>0:
            artifactlist=[148,150,152]
            #artifactlist=[123,134,153,154,155]
            if (catsegm_small.label[ct] in artifactlist):
                #print ('remove',catsegm_small.label[ct])
                segm_small.remove_labels(labels=catsegm_small.label[ct])
        
    #Make POM map and intensity map using each grism
    if k==0:
        directbkgsubbed_gr150c=data
        segm_all_gr150c=segm_small.data
        w=np.where(segm_large.data>0)
        segm_all_gr150c[w]=segm_large.data[w]
        pommap_all_gr150c=deepcopy(segm_all_gr150c)
        w=np.where(pommap_all_gr150c>0)
        pommap_all_gr150c[w]=1
        intens_all_gr150c=data*pommap_all_gr150c
    else:
        directbkgsubbed_gr150r=data
        segm_all_gr150r=segm_small.data
        w=np.where(segm_large.data>0)
        segm_all_gr150r[w]=segm_large.data[w]
        pommap_all_gr150r=deepcopy(segm_all_gr150r)
        w=np.where(pommap_all_gr150r>0)
        pommap_all_gr150r[w]=1
        intens_all_gr150r=data*pommap_all_gr150r

#Merge the GR150C and GR150R images
pommap=np.maximum(pommap_all_gr150c,pommap_all_gr150r)        
pomintens=np.maximum(intens_all_gr150c,intens_all_gr150r)

#Replace few pixels with intensity>1
w=np.where(pomintens>1.0)
pomintens[w]=1.0

#Find total "flux" lost due to accounted POM features
fractionfluxlost=np.sum(pomintens)/(2040.0*2040.0)
print ('total fraction of sensitivity decrease in accounted POM features=',fractionfluxlost)

#Output files
#According to Kevin's CV3 report JWST-STScI-004825 all filters have <0.5 pixel shift w.r.t. F115W except F200W. For F200W need to shift POM map by 1,2 pixels.
outfile1=outfile.replace('.fits','_pommap_F115W.fits')
outfile2=outfile.replace('.fits','_pomintens_F115W.fits')
outfile3=outfile.replace('.fits','_pommap_F200W.fits')
outfile4=outfile.replace('.fits','_pomintens_F200W.fits')

pommap_F115W=np.zeros(imshape)
pomintens_F115W=np.zeros(imshape)
pommap_F200W=np.zeros(imshape)
pomintens_F200W=np.zeros(imshape)

pommap_F115W[4:2044,4:2044]=pommap
pomintens_F115W[4:2044,4:2044]=pomintens
pommap_F200W[5:2045,2:2042]=pommap
pomintens_F200W[5:2045,2:2042]=pomintens

fits.writeto(outfile1, pommap_F115W, header=gr150cheader, overwrite=True)
fits.writeto(outfile2, pomintens_F115W, header=gr150cheader, overwrite=True)
fits.writeto(outfile3, pommap_F200W, header=gr150cheader, overwrite=True)
fits.writeto(outfile4, pomintens_F200W, header=gr150cheader, overwrite=True)


#Optionally output background-subtracted images, segmentation image and masked direct flat for checking
checkintermediateimages=True
if checkintermediateimages:
    directbkgsubbedimage_F115W_GR150C=np.zeros(imshape)
    directbkgsubbedimage_F115W_GR150C[4:2044,4:2044]=directbkgsubbed_gr150c
    checkoutfile='directbkgsubbedimage_F115W_GR150C.fits'
    fits.writeto(checkoutfile,directbkgsubbedimage_F115W_GR150C,gr150cheader,overwrite=True)
    directbkgsubbedimage_F115W_GR150R=np.zeros(imshape)
    directbkgsubbedimage_F115W_GR150R[4:2044,4:2044]=directbkgsubbed_gr150r
    checkoutfile='directbkgsubbedimage_F115W_GR150R.fits'
    fits.writeto(checkoutfile,directbkgsubbedimage_F115W_GR150R,gr150cheader,overwrite=True)
    segimage_F115W_GR150C=np.zeros(imshape)
    segimage_F115W_GR150C[4:2044,4:2044]=segm_all_gr150c
    checkoutfile='segimage_F115W_GR150C.fits'
    fits.writeto(checkoutfile,segimage_F115W_GR150C,gr150cheader,overwrite=True)
    segimage_F115W_GR150R=np.zeros(imshape)
    segimage_F115W_GR150R[4:2044,4:2044]=segm_all_gr150r
    checkoutfile='segimage_F115W_GR150R.fits'
    fits.writeto(checkoutfile,segimage_F115W_GR150R,gr150cheader,overwrite=True)
    directmasked=np.zeros(imshape)
    testdirectmasked=directactivedata-directactivedata*pommap
    directmasked[4:2044,4:2044]=testdirectmasked
    checkoutfile='directmasked_F115W.fits'
    fits.writeto(checkoutfile,directmasked,directheader,overwrite=True)

