#!/usr/bin/env python

#module to make dark current and noise images from darks, optionally using GDQ flags
#usage ./makedarknoisefilesgdq.py --cubedir= --slopedir={} --noisedir={} --outfileroot={} --usegdq=
#e.g. ./makedarknoisefilesgdq.py --cubedir=/mnt/jwstdata/cv3/colddarks/simcrs/third2runs/cubefixcr/ --slopedir=/mnt/jwstdata/cv3/colddarks/simcrs/third2runs/slopenocrrej/ --noisedir=/mnt/jwstdata/cv3/colddarks/simcrs/third2runs/testnoise/ --outfileroot=test --usegdq=True

import numpy as np
import re
import os
import optparse
import shlex, subprocess, signal
from astropy.io import fits
from astropy.stats import sigma_clip
import natsort
from copy import deepcopy
import datetime
import stsci.imagestats as imagestats
from jwst.datamodels import dqflags

# Command line options
op = optparse.OptionParser()
op.add_option("--cubedir")
op.add_option("--slopedir")
op.add_option("--noisedir")
op.add_option("--outfileroot")
op.add_option("--usegdq")

o, a = op.parse_args()
if a:
    print (sys.syserr, "unrecognized option: ",a)
    sys.exit(-1)

cubedir=o.cubedir
slopedir=o.slopedir
noisedir=o.noisedir
outfileroot=o.outfileroot
usegdq=o.usegdq
if usegdq == 'True' or usegdq == 'true':
    usegdq=True
else:
    usegdq=False

outputstats=True

if not os.path.exists(noisedir):
    os.makedirs(noisedir)

slope3d=[]
cdsstd3d=[]

dirlist=natsort.natsorted(os.listdir(slopedir))
dirlist[:] = (value for value in dirlist if value.startswith('NISNIRISS') and value.endswith('.fits'))
numimages=len(dirlist)
#reduce numimages for testing
#numimages=5

dirlistcube=natsort.natsorted(os.listdir(cubedir))
dirlistcube[:] = (value for value in dirlistcube if value.startswith('NISNIRISS') and value.endswith('.fits'))

for j in range(numimages):
    slopefile=os.path.join(slopedir,dirlist[j])
    hdulist=fits.open(slopefile)
    header=hdulist[0].header
    slope=hdulist['SCI'].data  
    cubefile=os.path.join(cubedir,dirlistcube[j])
    hdulist=fits.open(cubefile)
    cds=np.diff(np.squeeze(hdulist['SCI'].data),axis=0)
    cds1f=deepcopy(cds)
    numamp=4
    for k in range(numamp):
        ymin=k*512
        ymax=ymin+512
        medsec=np.median(cds1f[:,ymin:ymax,:],axis=1)
        medsecexpand = np.expand_dims(medsec, axis=1)
        medsecexpand=np.repeat(medsecexpand,512,axis=1)
        cds1f[:,ymin:ymax,:]-=medsecexpand

    cdsstd=np.std(cds,axis=0)     
    cds1fstd=np.std(cds1f,axis=0)     

    if usegdq==True:
        gdq=hdulist['GROUPDQ'].data
        crflagged=np.where(np.bitwise_and(gdq, dqflags.group['JUMP_DET']))
        crflag2d=np.zeros(slope.shape, dtype='uint8')
        crflag2d[crflagged[2],crflagged[3]]=1
    print (j,dirlist[j],dirlistcube[j])

    if len(slope3d)==0:
        slope3d=slope
        cdsstd3d=cdsstd
        cds1fstd3d=cds1fstd
        if usegdq==True:
            crflag3d=crflag2d
    else:
        slope3d=np.dstack((slope3d,slope))
        cdsstd3d=np.dstack((cdsstd3d,cdsstd))
        cds1fstd3d=np.dstack((cds1fstd3d,cds1fstd))
        if usegdq==True:
            crflag3d=np.dstack((crflag3d,crflag2d))

    del cdsstd,cds1fstd,cds,cds1f
    if usegdq==True:
        del crflag2d

print (slope3d.shape,cdsstd3d.shape)

#if more than 1/3 of the ramps of a pixel are flagged, unset all the flags
if usegdq==True:
    numrampsflagged=np.sum(crflag3d,2)
    w=np.where(numrampsflagged>(numimages/3.0))
    crflag3d[w[0],w[1],:]=0
    #optional output image of numrampsflagged to check
    #print (np.mean(numrampsflagged),numrampsflagged[np.where(numrampsflagged>(numimages/3.0))].size)
    #fits.writeto('testnumrampsflagged.fits',numrampsflagged.astype('uint16'),header,overwrite=True)
    

#sigma clipped arrays - iterate up to 20% of number of images to exclude cosmic rays
#sigiter=round(numimages/5.0)
#use this value if no simulated cosmic rays
sigiter=3

if usegdq==True:
    slope3dmasked=np.ma.masked_array(slope3d,mask=crflag3d)
    cdsstd3dmasked=np.ma.masked_array(cdsstd3d,mask=crflag3d)
    cds1fstd3dmasked=np.ma.masked_array(cds1fstd3d,mask=crflag3d)
else:
    slope3dmasked=slope3d
    cdsstd3dmasked=cdsstd3d
    cds1fstd3dmasked=cds1fstd3d

#Make and output dark current and slope noise images 
clippedslope3d=sigma_clip(slope3dmasked,sigma=3,maxiters=sigiter,axis=2)
clippedslope3dmean=np.ma.mean(clippedslope3d,axis=2)
clippedslope3dmedian=np.ma.median(clippedslope3d,axis=2)
clippedslope3dstd=np.ma.std(clippedslope3d,axis=2)

#Then do _pc_ version by median averaging of all 2d arrays 
clippedslope3dmedian1d=np.ma.median(clippedslope3d,axis=(0,1))
medclippedslope3dmedian1d=np.median(clippedslope3dmedian1d.data)
offsetclippedslope1d=medclippedslope3dmedian1d-clippedslope3dmedian1d

clippedslope3dpc=clippedslope3d+offsetclippedslope1d
clippedslope3dpcmean=np.ma.mean(clippedslope3dpc,axis=2)
clippedslope3dpcmedian=np.ma.median(clippedslope3dpc,axis=2)
clippedslope3dpcstd=np.ma.std(clippedslope3dpc,axis=2)

header['NAXIS'] = 2

clippedslope3dmean=clippedslope3dmean.data
clippedslope3dmedian=clippedslope3dmedian.data
clippedslope3dstd=clippedslope3dstd.data
clippedslope3dpcmean=clippedslope3dpcmean.data
clippedslope3dpcmedian=clippedslope3dpcmedian.data
clippedslope3dpcstd=clippedslope3dpcstd.data

meanfile=os.path.join(noisedir,outfileroot+'meandark.fits')
medianfile=meanfile.replace('meandark','mediandark')
stdfile=meanfile.replace('meandark','sigmadark')
fits.writeto(meanfile,clippedslope3dmean,header,overwrite=True)
fits.writeto(medianfile,clippedslope3dmedian,header,overwrite=True)
fits.writeto(stdfile,clippedslope3dstd,header,overwrite=True)

meanpcfile=os.path.join(noisedir,outfileroot+'meandarkzero.fits')
medianpcfile=meanpcfile.replace('meandarkzero','mediandarkzero')
stdpcfile=meanpcfile.replace('meandarkzero','sigmadarkzero')
fits.writeto(meanpcfile,clippedslope3dpcmean,header,overwrite=True)
fits.writeto(medianpcfile,clippedslope3dpcmedian,header,overwrite=True)
fits.writeto(stdpcfile,clippedslope3dpcstd,header,overwrite=True)

#optionally output stats for dark current and slope noise
if outputstats==True:

    print ('PC offsets: ',offsetclippedslope1d)
    print ('Std dev of PC offsets: ',np.std(offsetclippedslope1d))

    i = imagestats.ImageStats(clippedslope3dmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print (meanfile)
    i.printStats()
    i = imagestats.ImageStats(clippedslope3dmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print (medianfile)
    i.printStats()
    i = imagestats.ImageStats(clippedslope3dstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print (stdfile)
    i.printStats()
    i = imagestats.ImageStats(clippedslope3dpcmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print (meanpcfile)
    i.printStats()
    i = imagestats.ImageStats(clippedslope3dpcmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print (medianpcfile)
    i.printStats()
    i = imagestats.ImageStats(clippedslope3dpcstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print (stdpcfile)
    i.printStats()


#Make and output CDS noise images 
#do sigma clipping on CDS std stacks
clippedcdsstd3d=sigma_clip(cdsstd3dmasked,sigma=3,maxiters=sigiter,axis=2)
clippedcdsstd3dmean=np.ma.mean(clippedcdsstd3d,axis=2)
clippedcdsstd3dmedian=np.ma.median(clippedcdsstd3d,axis=2)
clippedcdsstd3dstd=np.ma.std(clippedcdsstd3d,axis=2)

clippedcdsstd3dmean=clippedcdsstd3dmean.data
clippedcdsstd3dmedian=clippedcdsstd3dmedian.data
clippedcdsstd3dstd=clippedcdsstd3dstd.data

clippedcds1fstd3d=sigma_clip(cds1fstd3dmasked,sigma=3,maxiters=sigiter,axis=2)
clippedcds1fstd3dmean=np.ma.mean(clippedcds1fstd3d,axis=2)
clippedcds1fstd3dmedian=np.ma.median(clippedcds1fstd3d,axis=2)
clippedcds1fstd3dstd=np.ma.std(clippedcds1fstd3d,axis=2)

clippedcds1fstd3dmean=clippedcds1fstd3dmean.data
clippedcds1fstd3dmedian=clippedcds1fstd3dmedian.data
clippedcds1fstd3dstd=clippedcds1fstd3dstd.data

#write output noise files
meanfile=os.path.join(noisedir,outfileroot+'meancdsstd.fits')
medianfile=meanfile.replace('meancdsstd','mediancdsstd')
stdfile=meanfile.replace('meancdsstd','sigmacdsstd')
fits.writeto(meanfile,clippedcdsstd3dmean,header,overwrite=True)
fits.writeto(medianfile,clippedcdsstd3dmedian,header,overwrite=True)
fits.writeto(stdfile,clippedcdsstd3dstd,header,overwrite=True)

mean1ffile=os.path.join(noisedir,outfileroot+'meancds1fstd.fits')
median1ffile=mean1ffile.replace('meancds1fstd','mediancds1fstd')
std1ffile=mean1ffile.replace('meancds1fstd','sigmacds1fstd')
fits.writeto(mean1ffile,clippedcds1fstd3dmean,header,overwrite=True)
fits.writeto(median1ffile,clippedcds1fstd3dmedian,header,overwrite=True)
fits.writeto(std1ffile,clippedcds1fstd3dstd,header,overwrite=True)

#optionally output stats for CDS noise
if outputstats==True:
    #Active pixel noise
    i = imagestats.ImageStats(clippedcdsstd3dmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print ('Non-reference pixels', meanfile)
    i.printStats()
    j = imagestats.ImageStats(clippedcdsstd3dmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print ('Non-reference pixels', medianfile)
    j.printStats()
    k = imagestats.ImageStats(clippedcdsstd3dstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print ('Non-reference pixels', stdfile)
    k.printStats()
    #1/f corrected active pixel noise
    i = imagestats.ImageStats(clippedcds1fstd3dmean[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print ('Non-reference pixels 1/f corrected', mean1ffile)
    i.printStats()
    j = imagestats.ImageStats(clippedcds1fstd3dmedian[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print ('Non-reference pixels 1/f corrected', median1ffile)
    j.printStats()
    k = imagestats.ImageStats(clippedcds1fstd3dstd[4:2044,4:2044],fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    print ('Non-reference pixels 1/f corrected', std1ffile)
    k.printStats()
