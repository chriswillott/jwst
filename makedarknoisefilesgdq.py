#!/usr/bin/env python

#module to make dark current and noise images from darks, optionally using GDQ flags
#usage ./makedarknoisefilesgdq.py cubedir slopedir noisedir outfileroot fraction_maxcr --sigiters sigiters --reffile_dark reffile_dark --usegdq --logstats
#e.g. ./makedarknoisefilesgdq.py /Users/willottc/niriss/detectors/cv3/colddarks/simcrs/cubefixcr/ /Users/willottc/niriss/detectors/cv3/colddarks/simcrs/rate/ /Users/willottc/niriss/detectors/cv3/colddarks/simcrs/noisewithdarksub/ test 0.33 --sigiters 3 --reffile_dark /Users/willottc/niriss/detectors/willott_reference_files/NIRISS_darkcube_cv3_38k_dms.fits --usegdq --logstats 

import numpy as np
import os
import argparse
from astropy.io import fits
from astropy.stats import sigma_clip
import glob
from copy import deepcopy
import datetime
import logging
import stsci.imagestats as imagestats
from jwst.datamodels import dqflags
import natsort

#===============================================================
# Command line arguments

parser = argparse.ArgumentParser()
parser.add_argument("cubedir", help="Location of dark cube images")
parser.add_argument("slopedir", help="Location of slope or rate images")
parser.add_argument("noisedir", help="Location of outout noise images")
parser.add_argument("outfileroot", help="Leading string of output noise filenames")
parser.add_argument("fraction_maxcr", type=float,  help="If more than fraction_maxcr of the exposures of a pixel are cosmic-ray flagged, unset all the flags because these cannot all be due to cosmic rays and it must be a noisy pixel")
parser.add_argument("--sigiters", type=int, default=3, help="Number of sigma-clipping iterations")
parser.add_argument("--reffile_dark", default=None, help="Dark current reference file for doing dark subtraction step")
parser.add_argument("--usegdq", help="Set if using GROUPDQ flagging", action="store_true")
parser.add_argument("--logstats", help="Set if want to log output statistics to a file", action="store_true")

args = parser.parse_args()

cubedir = args.cubedir
slopedir = args.slopedir
noisedir = args.noisedir
outfileroot = args.outfileroot
fraction_maxcr = args.fraction_maxcr
sigiters = args.sigiters
reffile_dark = args.reffile_dark
usegdq = args.usegdq
logstats = args.logstats

#===============================================================
#Make results directory if does not exist 
if not os.path.exists(noisedir):
    os.makedirs(noisedir)

#Log statistics after making dark current and noise files if flag is set  
if logstats == True:
    
        #Set up logging    
        logfile = datetime.datetime.now().strftime('darknoise_%Y%m%d_%H%M%S.log')
        logdirfile = os.path.join(noisedir,logfile)
        print ('makedarknoisefilesgdq log is {}'.format(logdirfile))

        logging.basicConfig(filename=logdirfile, filemode='w', level=logging.INFO, force=True)
        logging.info('Running makedarknoisefilesgdq.py with parameters:')
        logging.info('cubedir = {}'.format(cubedir))
        logging.info('slopedir = {}'.format(slopedir))
        logging.info('noisedir = {}'.format(noisedir))
        logging.info('outfileroot = {}'.format(outfileroot))
        logging.info('fraction_maxcr = {}'.format(fraction_maxcr))
        logging.info('sigiters = {}'.format(sigiters))
        logging.info('reffile_dark = {}'.format(reffile_dark))
        logging.info('usegdq = {}'.format(usegdq))
        
slope3d = []
cdsstd3d = []

#Get lists of all rate (slope) images from slopedir and cube files (after pipeline linearity step)
slopedirlist = natsort.natsorted(os.listdir(slopedir))
slopedirlist[:] = (value for value in slopedirlist if value.endswith('rate.fits'))
cubedirlist = natsort.natsorted(os.listdir(cubedir))
cubedirlist[:] = (value for value in cubedirlist if value.endswith('.fits'))
numimages = len(slopedirlist)
#reduce numimages for testing
#numimages = 5
print (numimages)

#Load the dark reference file data extension if using
if reffile_dark is not None:
    with fits.open(reffile_dark,memmap=True) as hdudarkref:
        dc = hdudarkref['SCI'].data

#===============================================================
#Loop over exposures
for j in range(numimages):
    slopefile=os.path.join(slopedir,slopedirlist[j])
    cubefile=os.path.join(cubedir,cubedirlist[j])
    #print ('Now running on j={} {} {}'.format(j,slopefile,cubefile))
    #logging.info('Now running on j={} {} {}'.format(j,slopefile,cubefile))
    with fits.open(slopefile) as slopehdulist:
        header = slopehdulist[0].header
        slope = slopehdulist['SCI'].data
    numgroups = header['NGROUPS']
    #Trim dark current reference file to match same number of groups
    if reffile_dark is not None:
        dc = dc[:numgroups,:,:]
    with fits.open(cubefile) as hdulist:
        #Assume each dark exposure contains only one integration, so strip off that axis
        cube = np.squeeze(hdulist['SCI'].data)
        #If dark current reference file provided first subtract the dark currrent off each group
        if reffile_dark is not None:
           cube -= dc 
        #Take cube differences to get CDS noise 
        cds = np.diff(cube,axis=0)
        cds1f = deepcopy(cds)
        #Loop over each amp separately to separate 1/f from other readnoise 
        numamp = 4
        for k in range(numamp):
            ymin = k*512
            ymax = ymin+512
            #Determine 1/f as median of each column
            medsec = np.median(cds1f[:,ymin:ymax,:],axis=1)
            medsecexpand = np.expand_dims(medsec, axis=1)
            medsecexpand = np.repeat(medsecexpand,512,axis=1)
            #Make CDS cube with 1/f noise subtracted
            cds1f[:,ymin:ymax,:] -= medsecexpand

        #Get std deviation of CDS cubes     
        cdsstd = np.std(cds,axis=0)     
        cds1fstd = np.std(cds1f,axis=0)     

        #If using GroupDQ flagged data find cosmic ray-flagged groups and make a 2D array of all cosmic ray hit pixels
        if usegdq == True:
            gdq = hdulist['GROUPDQ'].data
            crflagged = np.where(np.bitwise_and(gdq, dqflags.group['JUMP_DET']))
            crflag2d = np.zeros(slope.shape, dtype='uint8')
            crflag2d[crflagged[2],crflagged[3]] = 1

    #Stack slope and CDS arrays from all exposures    
    if len(slope3d) == 0:
        slope3d = slope
        cdsstd3d = cdsstd
        cds1fstd3d = cds1fstd
        if usegdq ==True:
            crflag3d = crflag2d
    else:
        slope3d = np.dstack((slope3d,slope))
        cdsstd3d = np.dstack((cdsstd3d,cdsstd))
        cds1fstd3d = np.dstack((cds1fstd3d,cds1fstd))
        if usegdq == True:
            crflag3d = np.dstack((crflag3d,crflag2d))

    #Delete arrays that won't be used again to save space        
    del cdsstd,cds1fstd,cds,cds1f
    if usegdq == True:
        del crflag2d

    

#if more than some fraction of the exposures of a pixel are cosmic-ray flagged, unset all the flags because these cannot all be due to cosmic rays and it must be a noisy pixel
if usegdq == True:
    numrampsflagged = np.sum(crflag3d,2)
    w = np.where(numrampsflagged>int(numimages*fraction_maxcr))
    crflag3d[w[0],w[1],:] = 0
    #optional output image of numrampsflagged to check
    #print (np.mean(numrampsflagged),numrampsflagged[np.where(numrampsflagged>(numimages/3.0))].size)
    #fits.writeto('testnumrampsflagged.fits',numrampsflagged.astype('uint16'),header,overwrite=True)
    
if usegdq == True:
    slope3dmasked = np.ma.masked_array(slope3d,mask=crflag3d)
    cdsstd3dmasked = np.ma.masked_array(cdsstd3d,mask=crflag3d)
    cds1fstd3dmasked = np.ma.masked_array(cds1fstd3d,mask=crflag3d)
else:
    slope3dmasked = slope3d
    cdsstd3dmasked = cdsstd3d
    cds1fstd3dmasked = cds1fstd3d

#===============================================================
#Make dark current and slope noise images 
clippedslope3d = sigma_clip(slope3dmasked,sigma=3,maxiters=sigiters,axis=2)
clippedslope3dmean = np.ma.mean(clippedslope3d,axis=2)
clippedslope3dmedian = np.ma.median(clippedslope3d,axis=2)
clippedslope3dstd = np.ma.std(clippedslope3d,axis=2)

#Make pedestal corrected version of slope images by median averaging of all 2d arrays 
clippedslope3dmedian1d = np.ma.median(clippedslope3d,axis=(0,1))
medclippedslope3dmedian1d = np.median(clippedslope3dmedian1d.data)
offsetclippedslope1d = medclippedslope3dmedian1d-clippedslope3dmedian1d

#Make pedestal corrected dark current and slope noise images 
clippedslope3dpc = clippedslope3d+offsetclippedslope1d
clippedslope3dpcmean = np.ma.mean(clippedslope3dpc,axis=2)
clippedslope3dpcmedian = np.ma.median(clippedslope3dpc,axis=2)
clippedslope3dpcstd = np.ma.std(clippedslope3dpc,axis=2)

header['NAXIS'] = 2

clippedslope3dmean = clippedslope3dmean.data
clippedslope3dmedian = clippedslope3dmedian.data
clippedslope3dstd = clippedslope3dstd.data
clippedslope3dpcmean = clippedslope3dpcmean.data
clippedslope3dpcmedian = clippedslope3dpcmedian.data
clippedslope3dpcstd = clippedslope3dpcstd.data

#Write dark current and slope noise output files
meanfile = os.path.join(noisedir,outfileroot+'meandark.fits')
medianfile = meanfile.replace('meandark','mediandark')
stdfile = meanfile.replace('meandark','sigmadark')
fits.writeto(meanfile,clippedslope3dmean,header,overwrite=True)
fits.writeto(medianfile,clippedslope3dmedian,header,overwrite=True)
fits.writeto(stdfile,clippedslope3dstd,header,overwrite=True)

meanpcfile = os.path.join(noisedir,outfileroot+'meandarkzero.fits')
medianpcfile = meanpcfile.replace('meandarkzero','mediandarkzero')
stdpcfile = meanpcfile.replace('meandarkzero','sigmadarkzero')
fits.writeto(meanpcfile,clippedslope3dpcmean,header,overwrite=True)
fits.writeto(medianpcfile,clippedslope3dpcmedian,header,overwrite=True)
fits.writeto(stdpcfile,clippedslope3dpcstd,header,overwrite=True)

#===============================================================
#Make and output CDS noise images 
#do sigma clipping on CDS std stacks
clippedcdsstd3d = sigma_clip(cdsstd3dmasked,sigma=3,maxiters=sigiters,axis=2)
clippedcdsstd3dmean = np.ma.mean(clippedcdsstd3d,axis=2)
clippedcdsstd3dmedian = np.ma.median(clippedcdsstd3d,axis=2)
clippedcdsstd3dstd = np.ma.std(clippedcdsstd3d,axis=2)

clippedcdsstd3dmean = clippedcdsstd3dmean.data
clippedcdsstd3dmedian = clippedcdsstd3dmedian.data
clippedcdsstd3dstd = clippedcdsstd3dstd.data

clippedcds1fstd3d = sigma_clip(cds1fstd3dmasked,sigma=3,maxiters=sigiters,axis=2)
clippedcds1fstd3dmean = np.ma.mean(clippedcds1fstd3d,axis=2)
clippedcds1fstd3dmedian = np.ma.median(clippedcds1fstd3d,axis=2)
clippedcds1fstd3dstd = np.ma.std(clippedcds1fstd3d,axis=2)

clippedcds1fstd3dmean = clippedcds1fstd3dmean.data
clippedcds1fstd3dmedian = clippedcds1fstd3dmedian.data
clippedcds1fstd3dstd = clippedcds1fstd3dstd.data

#write output CDS noise files
meancdsstdfile = os.path.join(noisedir,outfileroot+'meancdsstd.fits')
mediancdsstdfile = meancdsstdfile.replace('meancdsstd','mediancdsstd')
stdcdsstdfile = meancdsstdfile.replace('meancdsstd','sigmacdsstd')
fits.writeto(meancdsstdfile,clippedcdsstd3dmean,header,overwrite=True)
fits.writeto(mediancdsstdfile,clippedcdsstd3dmedian,header,overwrite=True)
fits.writeto(stdcdsstdfile,clippedcdsstd3dstd,header,overwrite=True)

mean1fcdsstdfile = os.path.join(noisedir,outfileroot+'meancds1fstd.fits')
median1fcdsstdfile = mean1fcdsstdfile.replace('meancds1fstd','mediancds1fstd')
std1fcdsstdfile = mean1fcdsstdfile.replace('meancds1fstd','sigmacds1fstd')
fits.writeto(mean1fcdsstdfile,clippedcds1fstd3dmean,header,overwrite=True)
fits.writeto(median1fcdsstdfile,clippedcds1fstd3dmedian,header,overwrite=True)
fits.writeto(std1fcdsstdfile,clippedcds1fstd3dstd,header,overwrite=True)

#===============================================================
#optionally log stats
if logstats == True:

    logging.info('Pedestal Correction offsets: {}'.format(offsetclippedslope1d))
    logging.info('Std dev of Pedestal Correction offsets: {}'.format(np.std(offsetclippedslope1d)))
    
    logging.info('')
    logging.info('Stats of slope and slope noise files, with and without predestal correction')

    i = imagestats.ImageStats(clippedslope3dmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(meanfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedslope3dmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(medianfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedslope3dstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(stdfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedslope3dpcmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(meanpcfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedslope3dpcmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(medianpcfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedslope3dpcstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(stdpcfile,i.mean,i.median,i.stddev))

    logging.info('')
    logging.info('Stats of CDS noise files (non-reference pixels), with and without 1/f subtraction')
    i = imagestats.ImageStats(clippedcdsstd3dmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(meancdsstdfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedcdsstd3dmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(mediancdsstdfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedcdsstd3dstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('{}  Mean: {}, Median: {}, Std dev: {}'.format(stdcdsstdfile,i.mean,i.median,i.stddev))
    #1/f subtracted CDS noise
    i = imagestats.ImageStats(clippedcds1fstd3dmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('1/f corrected {}  Mean: {}, Median: {}, Std dev: {}'.format(mean1fcdsstdfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedcds1fstd3dmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('1/f corrected {}  Mean: {}, Median: {}, Std dev: {}'.format(median1fcdsstdfile,i.mean,i.median,i.stddev))
    i = imagestats.ImageStats(clippedcds1fstd3dstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('1/f corrected {}  Mean: {}, Median: {}, Std dev: {}'.format(std1fcdsstdfile,i.mean,i.median,i.stddev))
