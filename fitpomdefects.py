#!/usr/bin/env python

#This procedure takes in direct and dispersed NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror, including the coronagraphic spots. The two outputs are a mask map of the features and an image of transmission due to the features. The detection direct image flat is usually F115W because this is best to detect the features. Due to the wavelength-dependence of flux loss the filter of the measure direct image flat is usually the output filter. Note must inspect and adjust artifacts if detectflat is not CV3 F115W+CLEAR flat ./slope/NIST74-F115-FL-6003020347_1_496_SE_2016-01-03T02h19m52_slope_norm.fits.

#Usage, e.g.
#fitpomdefects.py --detectflat='./slope/NIST74-F115-FL-6003020347_1_496_SE_2016-01-03T02h19m52_slope_norm.fits' --measureflat='./slope/NIST74-F150-FL-6003023303_1_496_SE_2016-01-03T02h46m02_slope_norm.fits' --gr150cflat='./slope/NIST74-F115-C-FL-6003044302_1_496_SE_2016-01-03T04h57m06_slope_norm.fits' --gr150rflat='./slope/NIST74-F115-R-FL-6003061354_1_496_SE_2016-01-03T06h29m13_slope_norm.fits' --mask='/Users/willottc/niriss/detectors/willott_reference_files/nocrs/jwst_niriss_cv3_38k_nocrs_bpm_minimal.fits'  --outname='/Users/willottc/niriss/detectors/willott_reference_files/jwst_niriss_cv3_F150W.fits' --ngrow=1

import numpy as np
from scipy import ndimage
import optparse
import os, os.path
import astropy.io.fits as fits
from astropy.stats import SigmaClip
from copy import deepcopy
from photutils import detect_threshold,detect_sources,Background2D, MedianBackground,source_properties, CircularAperture, CircularAnnulus
from jwst.datamodels import dqflags

# Command line options
op = optparse.OptionParser()
op.add_option("--detectflat")
op.add_option("--measureflat")
op.add_option("--gr150cflat")
op.add_option("--gr150rflat")
op.add_option("--mask")
op.add_option("--outname")
op.add_option("--ngrow")

o, a = op.parse_args()
if a:
    print (sys.syserr, "unrecognized option: ",a)
    sys.exit(-1)

detectflatfile=o.detectflat
measureflatfile=o.measureflat
gr150cflatfile=o.gr150cflat
gr150rflatfile=o.gr150rflat
maskfile=o.mask
outname=o.outname
ngrow=o.ngrow
if ngrow==None:
    ngrow=1

#Read in flat field data - exclude reference pixels
hdulist1=fits.open(detectflatfile)
detectheader=hdulist1['PRIMARY'].header
detectactivedata=hdulist1['SCI'].data[4:2044,4:2044]
hdulist2=fits.open(measureflatfile)
measureheader=hdulist2['PRIMARY'].header
measureactivedata=hdulist2['SCI'].data[4:2044,4:2044]
hdulist3=fits.open(gr150cflatfile)
gr150cpriheader=hdulist3['PRIMARY'].header
gr150csciheader=hdulist3['SCI'].header
gr150cactivedata=hdulist3['SCI'].data[4:2044,4:2044]
hdulist4=fits.open(gr150rflatfile)
gr150ractivedata=hdulist4['SCI'].data[4:2044,4:2044]
imshape=hdulist1['SCI'].data.shape

#load bad pixel file
hdulist=fits.open(maskfile)
maskdata=hdulist['DQ'].data
donotuseindices=np.where(np.bitwise_and(maskdata, dqflags.pixel['DO_NOT_USE']))
numbadpix=donotuseindices[0].size

#Get values to normalise flats
normalizedetect=np.median(detectactivedata)
normalizemeasure=np.median(measureactivedata)
normalizegr150c=np.median(gr150cactivedata)
normalizegr150r=np.median(gr150ractivedata)

#According to Kevin Volk's CV3 report JWST-STScI-004825 all filters have <0.5 pixel shift w.r.t. F115W except F200W.
#For F200W need to shift POM map by 1,2 pixels.
filterout=measureheader['PUPIL']
if filterout=='F200W':
    xoff=-2
    yoff=1
else:    
    xoff=0
    yoff=0
#F115W positions of coronagraphic spots and radii defining the dark central regions 
coro1cen=np.array([1061+xoff,1884+yoff])
coro2cen=np.array([750+xoff,1860.45+yoff])
coro3cen=np.array([439.45+xoff,1837.45+yoff])
coro4cen=np.array([129+xoff,1814.45+yoff])
coro1cenfloor=np.floor(coro1cen).astype('int')
coro2cenfloor=np.floor(coro2cen).astype('int')
coro3cenfloor=np.floor(coro3cen).astype('int')
coro4cenfloor=np.floor(coro4cen).astype('int')
#Reduce radii that define the complete spot region to just cover dark central region
coro1radius=round(19*0.65)
coro2radius=round(14*0.62)
coro3radius=round(8*0.50)
coro4radius=round(6*0.48)

#Subtract detect off grism flats, so POM features in detect appear positive 
gr150cminusdetect=gr150cactivedata/normalizegr150c-detectactivedata/normalizedetect
gr150rminusdetect=gr150ractivedata/normalizegr150r-detectactivedata/normalizedetect

#Subtract measure off grism flats, so POM features in measure appear positive 
gr150cminusmeasure=gr150cactivedata/normalizegr150c-measureactivedata/normalizemeasure
gr150rminusmeasure=gr150ractivedata/normalizegr150r-measureactivedata/normalizemeasure

#Initialize full size output arrays
pommap=np.zeros(imshape)
pomintens=np.zeros(imshape)

#Iterate over two grisms
grismminusdetectdata=[gr150cminusdetect,gr150rminusdetect]
grismminusmeasuredata=[gr150cminusmeasure,gr150rminusmeasure]
for k in range(2):
        
    #Subtract a background from grismminusdetectdata
    #Don't need a mask excluding low and high pixels as sigma clipping will exclude bright and dark areas
    sigma_clip = SigmaClip(sigma=3., maxiters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(grismminusdetectdata[k], (24, 24),filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    grismminusdetectdata[k]=grismminusdetectdata[k]-bkg.background
    
    #Replace all highly negative pixels (these are dispersed POM features) with zeros
    v=np.where(grismminusdetectdata[k]<-0.02)
    grismminusdetectdata[k][v]=0.0

    #Run source extraction twice
    #Do not use a filter as would make single bright pixels appear as sources
    #Do first run to find small things 
    threshold_small = detect_threshold(grismminusdetectdata[k], snr=3.0)
    segm_small = detect_sources(grismminusdetectdata[k], threshold_small, npixels=4, connectivity=4, filter_kernel=None) 
    #Do second run to find large things with lower significance and merge the two
    threshold_large = detect_threshold(grismminusdetectdata[k], snr=1.5)
    segm_large = detect_sources(grismminusdetectdata[k], threshold_large, npixels=100, connectivity=8, filter_kernel=None) 

    #Remove some sources from segmentation image
    catsegm_large = source_properties(grismminusdetectdata[k], segm_large)
    numsource=np.array(catsegm_large.label).size
    for ct in range(numsource):
        #Remove bad detector region at x=12, y=1536
        if ((catsegm_large.ycentroid[ct].value>1520)&(catsegm_large.ycentroid[ct].value<1555)&(catsegm_large.xcentroid[ct].value<25)):
            #print ('remove',catsegm_large.label[ct])
            segm_large.remove_labels(labels=catsegm_large.label[ct])
 
    #Remove some sources from segmentation image
    catsegm_small = source_properties(grismminusdetectdata[k], segm_small)
    numsource=np.array(catsegm_small.label).size
    for ct in range(numsource):
        print (ct,catsegm_small.label[ct],catsegm_small.xcentroid.value[ct],catsegm_small.ycentroid.value[ct],catsegm_small.area.value[ct]) 
 
        #Remove all sources in lower 100 pixel strip
        if (catsegm_small.ycentroid[ct].value<100.0):
            #print ('remove',catsegm_small.label[ct])
            segm_small.remove_labels(labels=catsegm_small.label[ct])
        #Remove small sources (area<6 pixels) except in top part of detector because POM not in focus so can't be real. 
        elif ((catsegm_small.ycentroid[ct].value>100.0 and catsegm_small.ycentroid[ct].value<1700.0 and catsegm_small.area[ct].value<6)):
            #print ('remove',catsegm_small.label[ct])
            segm_small.remove_labels(labels=catsegm_small.label[ct])
        #Remove bad detector region at x=12, y=1536
        elif ((catsegm_small.ycentroid[ct].value>1520)&(catsegm_small.ycentroid[ct].value<1555)&(catsegm_small.xcentroid[ct].value<25)):
            #print ('remove',catsegm_small.label[ct])
            segm_small.remove_labels(labels=catsegm_small.label[ct])

        #Remove few artifacts using segemtation IDs - note IDs will change if grismminusdetectdata or software changes
        #F115-FL-sub-F115-C-FL  threshold = detect_threshold(grismminusdetectdata[k], snr=3.0 - no artifacts

        #F115-FL-sub-F115-R-FL  threshold = detect_threshold(grismminusdetectdata[k], snr=3.0 - 3 artifacts
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
                
    #Subtract a background from grismminusmeasuredata
    #Don't need a mask excluding low and high pixels as sigma clipping will exclude bright and dark areas
    sigma_clip = SigmaClip(sigma=3., maxiters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(grismminusmeasuredata[k], (24, 24),filter_size=(5, 5), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    grismminusmeasuredata[k]=grismminusmeasuredata[k]-bkg.background
    
    #Replace all negative pixels (these are dispersed POM features and noise) with zeros
    v=np.where(grismminusmeasuredata[k]<0.0)
    grismminusmeasuredata[k][v]=0.0
    
    #Make POM map and intensity map using this grism
    segm2040=segm_small.data
    w=np.where(segm_large.data>0)
    segm2040[w]=segm_large.data[w]
    pommap2040=deepcopy(segm2040)
    w=np.where(pommap2040>0)
    pommap2040[w]=1
    #Grow detection region
    growarray=2*int(ngrow)+1
    kern = np.ones((growarray,growarray))
    #kern = np.array([[ngrow,ngrow,ngrow],[ngrow,ngrow,ngrow],[ngrow,ngrow,ngrow]])
    pommap2040=ndimage.convolve(pommap2040, kern, mode='constant', cval=0.0)
    pommap2040[np.where(pommap2040>1.0)]=1.0
    
    #Merge the GR150C and GR150R images and shift if necessary when placing within the full detector
    if k==0:
        pommap[4+yoff:2044+yoff,4+xoff:2044+xoff]=pommap2040
        pomintens[4:2044,4:2044]=grismminusmeasuredata[k]*pommap[4:2044,4:2044]
    else:
        pommap[4+yoff:2044+yoff,4+xoff:2044+xoff]=np.maximum(pommap[4+yoff:2044+yoff,4+xoff:2044+xoff],pommap2040)        
        pomintens[4:2044,4:2044]=np.maximum(pomintens[4:2044,4:2044],(grismminusmeasuredata[k]*pommap[4:2044,4:2044]))

#set constant high value in central regions of coronagraphic spots as too noisy to measure in flats 
coro1aperture = CircularAperture(coro1cen, r=coro1radius)
coro1_obj_mask = coro1aperture.to_mask(method='center')
coro1_obj_mask_corr=2.0*coro1_obj_mask.data
pomintens[coro1cenfloor[1]-coro1radius:coro1cenfloor[1]+coro1radius+1,coro1cenfloor[0]-coro1radius:coro1cenfloor[0]+coro1radius+1]+=coro1_obj_mask_corr
coro2aperture = CircularAperture(coro2cen, r=coro2radius)
coro2_obj_mask = coro2aperture.to_mask(method='center')
coro2_obj_mask_corr=2.0*coro2_obj_mask.data
pomintens[coro2cenfloor[1]-coro2radius:coro2cenfloor[1]+coro2radius+1,coro2cenfloor[0]-coro2radius:coro2cenfloor[0]+coro2radius+1]+=coro2_obj_mask_corr
coro3aperture = CircularAperture(coro3cen, r=coro3radius)
coro3_obj_mask = coro3aperture.to_mask(method='center')
coro3_obj_mask_corr=2.0*coro3_obj_mask.data
pomintens[coro3cenfloor[1]-coro3radius:coro3cenfloor[1]+coro3radius+1,coro3cenfloor[0]-coro3radius:coro3cenfloor[0]+coro3radius+1]+=coro3_obj_mask_corr
coro4aperture = CircularAperture(coro4cen, r=coro4radius)
coro4_obj_mask = coro4aperture.to_mask(method='center')
coro4_obj_mask_corr=2.0*coro4_obj_mask.data
pomintens[coro4cenfloor[1]-coro4radius:coro4cenfloor[1]+coro4radius+1,coro4cenfloor[0]-coro4radius:coro4cenfloor[0]+coro4radius+1]+=coro4_obj_mask_corr

#Replace pixels with intensity>0.98 to 0.98 as that is minimum seen in spots in CV3
w=np.where(pomintens>0.98)
pomintens[w]=0.98

#Interpolate across bad pixels if at least 3 out of 4 corner neighbours are in POM mask
pomintensfixbadpix=deepcopy(pomintens)
for j in range(numbadpix):
    y=donotuseindices[0][j]
    x=donotuseindices[1][j]
    #Do not include reference pixels
    if y>3 and y<2044 and x>3 and x<2044:
        neighborsumpommap=pommap[y-1,x-1]+pommap[y-1,x+1]+pommap[y+1,x-1]+pommap[y+1,x+1]
        if neighborsumpommap>2:
            pomintensfixbadpix[y,x]=np.median([pomintens[y-1,x-1],pomintens[y-1,x+1],pomintens[y+1,x-1],pomintens[y+1,x+1]])

#Find total "flux" lost due to accounted POM features
fractionfluxlost=np.sum(pomintensfixbadpix)/(2040.0*2040.0)
print ('total fraction of sensitivity decrease in accounted POM features=',fractionfluxlost)

#Will output intensity files as transmission so do 1-intens
pomtransmission=1.0-pomintensfixbadpix

outfile1=outname.replace('.fits','_pommap.fits')
outfile2=outname.replace('.fits','_pomtransmission.fits')

#Add information about inputs to output file headers
gr150cpriheader.append(('', ''),end=True)
gr150cpriheader.append(('', 'POM Fitting Information'),end=True)
gr150cpriheader.append(('', ''),end=True)
gr150cpriheader.append(('DETPOM',os.path.basename(detectflatfile),'Direct flat for POM detection'),end=True)
gr150cpriheader.append(('MEASPOM',os.path.basename(measureflatfile),'Direct flat for POM measurement'),end=True)
gr150cpriheader.append(('CFLATPOM',os.path.basename(gr150cflatfile),'GR150C flat for POM'),end=True)
gr150cpriheader.append(('RFLATPOM',os.path.basename(gr150rflatfile),'GR150R flat for POM'),end=True)
gr150cpriheader.append(('BPMPOM',os.path.basename(maskfile),'Bad pixel mask for POM'),end=True)
gr150cpriheader.append(('NGROWPOM',ngrow,'Number of pixels to grow POM features'),end=True)

#Output files
print ('Writing',outfile1)
hdup=fits.PrimaryHDU(data=None,header=gr150cpriheader)
hdusci=fits.ImageHDU(data=pommap,header=gr150csciheader,name='SCI')
hdulistout=fits.HDUList([hdup,hdusci])
hdulistout.writeto(outfile1,overwrite=True)

print ('Writing',outfile2)
hdup=fits.PrimaryHDU(data=None,header=gr150cpriheader)
hdusci=fits.ImageHDU(data=pomtransmission,header=gr150csciheader,name='SCI')
hdulistout=fits.HDUList([hdup,hdusci])
hdulistout.writeto(outfile2,overwrite=True)


