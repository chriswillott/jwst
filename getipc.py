#!/usr/bin/env python
#This procedure determines a 5x5 IPC kernel image based on spread of charge from hot pixels
#Includes the NIRISS void map to calculate a different IPC kernel within the void region 
#Uses noise and bad pixel files output by makebpm.py
#Outputs 5x5 images for each amp in the void and outside and a 5x5x2048x2048 array of IPC across the full detector

from __future__ import print_function
import numpy as np
import re
import os
import optparse
import shlex, subprocess, signal
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table, Column
import natsort
from copy import deepcopy
import time
import stsci.imagestats as imagestats
from photutils import CircularAnnulus
from photutils import aperture_photometry
import scipy.ndimage

#Set to enforce IPC symmetry rather than use each of the 25 pixels independently
makesymm=True

#NIRISS Detector 
#noise files are in units of ADU so need gain to convert to electrons
gain=1.62

#define reference file locations
refdir='/Users/willottc/niriss/detectors/willott_reference_files/'
longbpmfile=refdir+'jwst_niriss_cv3_38k_bpm_long.fits'
shortbpmfile=refdir+'jwst_niriss_cv3_38k_bpm_short.fits'
typebpmdir=refdir+'badpixeltypes/'
typebpmfile=typebpmdir+'jwst_niriss_cv3_38k_bpm_type'
hotfile=typebpmfile+'_hot.fits'
badflatfile=typebpmfile+'_unrelflat.fits'
donotuselongfile=typebpmfile+'_donotuse_long.fits'
refpixfile=typebpmfile+'_refpixel.fits'
voidmaskfile=refdir+'volk_niriss_voidmask9.fits'
ipcfile=refdir+'jwst_niriss_cv3_38k_measured_ipc.fits'

ipc4d=np.zeros((5,5,2048,2048))


#location of dark noise files
darknoisedir='/Users/willottc/niriss/detectors/cv3/colddarks/noise/'

#Load dark noise data and multiply by gain to get in e-
slopenoisefile=darknoisedir+'cv3_38k_sigmadarkzero.fits'
hdulist=fits.open(slopenoisefile)
header=hdulist[0].header
slopenoise=hdulist[0].data*gain

darkcurrfile=darknoisedir+'cv3_38k_mediandark.fits'
hdulist=fits.open(darkcurrfile)
darkcurr=hdulist[0].data*gain

cdsnoisefile=darknoisedir+'cv3_38k_mediancds1fstd.fits'
hdulist=fits.open(cdsnoisefile)
cdsnoise=hdulist[0].data*gain


#load hot pixel array
hdulist=fits.open(hotfile)
hotflagged=hdulist[0].data

#load bad flat to exclude pixels bordering on ref pixels
hdulist=fits.open(badflatfile)
badflatflagged=hdulist[0].data

#load bad pixel file and refpixel file to exclude bad and ref pixels from median DC
hdulist=fits.open(donotuselongfile)
donotuselongflagged=hdulist[0].data
hdulist=fits.open(refpixfile)
refpixflagged=hdulist[0].data

#load voidmask
hdulist=fits.open(voidmaskfile)
voidmask=hdulist[0].data

#create an empty array for bad pixel mask and donotuse mask
arrshape=(5,5)
ipc=np.zeros(arrshape)


##Dark images section##
#slopenoiseactive=slopenoise[4:2044,4:2044]
#imstatslope = imagestats.ImageStats(slopenoiseactive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
#imstatslope.printStats()

#cdsnoiseactive=cdsnoise[4:2044,4:2044]
#imstatcds = imagestats.ImageStats(cdsnoiseactive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
#imstatcds.printStats()

#darkcurractive=darkcurr[4:2044,4:2044]
imstatdark = imagestats.ImageStats(darkcurr,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatdark.printStats()

#Get hot pixels not in void - need to add void mask sel below
w=(np.where((voidmask!=1)&(hotflagged==1)&(badflatflagged!=1)&(darkcurr>1.0)&(darkcurr<100.0)))
print (darkcurr[w].size)
yhotnotvoid=w[0]
xhotnotvoid=w[1]
#print (yhotnotvoid,xhotnotvoid)
numhotnotvoid=yhotnotvoid.size

#Get hot pixels in void - repeat above block
w=(np.where((voidmask==1)&(hotflagged==1)&(badflatflagged!=1)&(darkcurr>1.0)&(darkcurr<100.0)))
print (darkcurr[w].size)
yhotinvoid=w[0]
xhotinvoid=w[1]
#print (yhotinvoid,xhotinvoid)
numhotinvoid=yhotinvoid.size



#Use per amplifier - exclude pixels near amp edges 
amplifier=np.array(['4','3','2','1','all'])
colstart=np.array([7,514,1028,1538,7])
colstop=np.array([510,1020,1534,2041,2041])

for j in range(5):

    ampmask=np.zeros((2048,2048),'uint16')
    ampmask[j*512:(1+j)*512,:]=1
    #Get stats in amplifier section for medians in out and of voids
    darkcurrsection=darkcurr[colstart[j]:colstop[j],:]
    badflatflaggedsection=badflatflagged[colstart[j]:colstop[j],:]
    donotuselongflaggedsection=donotuselongflagged[colstart[j]:colstop[j],:]
    refpixflaggedsection=refpixflagged[colstart[j]:colstop[j],:]
    voidmasksection=voidmask[colstart[j]:colstop[j],:]

    #=====================================================================================================
    #Firstly work on out of void region
    w=(np.where((voidmasksection!=1)&(badflatflaggedsection!=1)&(donotuselongflaggedsection!=1)&(refpixflaggedsection!=1)))
    clippeddarkcurrsection=sigma_clip(darkcurrsection[w],sigma=3,maxiters=5)
    mediandarkcurrnotvoid=np.ma.median(clippeddarkcurrsection)
    #mediandarkcurrnotvoid=np.median(darkcurrsection[w])
    print (amplifier[j],mediandarkcurrnotvoid)
    #In void - not for amp A
    if j>0:
        w=(np.where((voidmasksection==1)&(badflatflaggedsection!=1)&(donotuselongflaggedsection!=1)&(refpixflaggedsection!=1)))
        mediandarkcurrinvoid=np.median(darkcurrsection[w])
        print (amplifier[j],mediandarkcurrinvoid)
        
    #only use pixels >3 away from ref pixels and with no hot (>1 e/s) neighbours
    fiveby3d=[]
    for k in range(numhotnotvoid): 
        if ((yhotnotvoid[k]>colstart[j])and(yhotnotvoid[k]<colstop[j])and(xhotnotvoid[k]>6)and(xhotnotvoid[k]<2041)):
            fivebydc=darkcurr[yhotnotvoid[k]-3:yhotnotvoid[k]+4,xhotnotvoid[k]-3:xhotnotvoid[k]+4]
            numhotincutout=fivebydc[np.where(fivebydc>1.0)].size
            if numhotincutout<2:
                fiveby=fivebydc-mediandarkcurrnotvoid
                fiveby=fiveby/np.sum(fiveby)
                #if doing all amps together flip if in amps 4 or 2
                if ((j==4) and (((yhotnotvoid[k]>colstart[0]) and (yhotnotvoid[k]<colstop[0]))or((yhotnotvoid[k]>colstart[2]) and (yhotnotvoid[k]<colstop[2])))):
                    fiveby=np.flip(fiveby,axis=0)
                
                if len(fiveby3d)==0:
                    fiveby3d=fiveby
                else:
                    fiveby3d=np.dstack((fiveby3d,fiveby))

    clippedfiveby3d=sigma_clip(fiveby3d,sigma=3,maxiters=5,axis=2)

    clippedfiveby3dmean=np.ma.mean(clippedfiveby3d,axis=2)
    #print ('clippedfiveby3dmean',clippedfiveby3dmean)
    clippedfiveby3dmedian=np.ma.median(clippedfiveby3d,axis=2)
    #print ('clippedfiveby3dmedian',clippedfiveby3dmedian)
    header['NAXIS'] = 2

    border=np.concatenate((np.ravel(clippedfiveby3dmedian[0,:]),np.ravel(clippedfiveby3dmedian[-1,:]),np.ravel(clippedfiveby3dmedian[1:6,0]),np.ravel(clippedfiveby3dmedian[1:6,-1])))
    normclippedfiveby3dmedian=clippedfiveby3dmedian.data-np.mean(border)
    normclippedfiveby3dmedian=normclippedfiveby3dmedian/np.sum(normclippedfiveby3dmedian)
    print (fiveby3d.shape,np.mean(border))

    #trim & enforce symmetry for some pixels
    normclippedfiveby3dmedian=normclippedfiveby3dmedian[1:6,1:6]
    mask1=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
    mask1[2,1]=1
    mask1[2,3]=1
    mask2=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
    mask2[1,1]=1
    mask2[1,3]=1
    mask2[3,1]=1
    mask2[3,3]=1
    mask3=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
    mask3[0,2]=1
    mask3[4,2]=1
    mask3[2,0]=1
    mask3[2,4]=1
    mask4=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
    mask4[0,1]=1
    mask4[0,3]=1
    mask4[1,0]=1
    mask4[1,4]=1
    mask4[3,0]=1
    mask4[3,4]=1
    mask4[4,1]=1
    mask4[4,3]=1
    mask5=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
    mask5[0,0]=1
    mask5[0,4]=1
    mask5[4,0]=1
    mask5[4,4]=1

    #set option of enforced symmetry 
    if makesymm==True:
        w=np.where(mask1==1)
        normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
        w=np.where(mask2==1)
        normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
        w=np.where(mask3==1)
        normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
        w=np.where(mask4==1)
        normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
        w=np.where(mask5==1)
        normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
    
    filemedian='ipc5by5median_amp%s_notvoid.fits' %  (amplifier[j])
    fits.writeto(filemedian,normclippedfiveby3dmedian,header,overwrite=True)

    print (normclippedfiveby3dmedian[2,1])
    
    if j<4:
        #put section in 4D IPC file
        w=np.where((voidmask!=1)&(ampmask==1))
        normclippedfiveby3dmedianexpand=np.expand_dims(normclippedfiveby3dmedian,axis=2)
        normclippedfiveby3dmedianexpand=np.expand_dims(normclippedfiveby3dmedianexpand,axis=3)
        normclippedfiveby3dmedianexpand=np.repeat(normclippedfiveby3dmedianexpand,2048,axis=2)
        normclippedfiveby3dmedianexpand=np.repeat(normclippedfiveby3dmedianexpand,2048,axis=3)
        ipc4d[:,:,w[0],w[1]]=normclippedfiveby3dmedianexpand[:,:,w[0],w[1]]

    #=====================================================================================================
    #then repeat this section for void - not for amp A and for amp D too few hot pixels so use inverted amp C
    if j==1 or j==2 or j==4:
        fiveby3d=[]
        for k in range(numhotinvoid): 
            if ((yhotinvoid[k]>colstart[j])and(yhotinvoid[k]<colstop[j])and(xhotinvoid[k]>6)and(xhotinvoid[k]<2041)):
                fivebydc=darkcurr[yhotinvoid[k]-3:yhotinvoid[k]+4,xhotinvoid[k]-3:xhotinvoid[k]+4]
                numhotincutout=fivebydc[np.where(fivebydc>1.0)].size
                if numhotincutout<2:
                    fiveby=fivebydc-mediandarkcurrinvoid
                    fiveby=fiveby/np.sum(fiveby)
                    #if doing all amps together flip if in amps 4 or 2
                    if ((j==4) and (((yhotnotvoid[k]>colstart[0]) and (yhotnotvoid[k]<colstop[0]))or((yhotnotvoid[k]>colstart[2]) and (yhotnotvoid[k]<colstop[2])))):
                        fiveby=np.flip(fiveby,axis=0)
                    #print (fiveby.shape)
                    if len(fiveby3d)==0:
                        fiveby3d=fiveby
                    else:
                        fiveby3d=np.dstack((fiveby3d,fiveby))

        clippedfiveby3d=sigma_clip(fiveby3d,sigma=3,maxiters=5,axis=2)

        clippedfiveby3dmean=np.ma.mean(clippedfiveby3d,axis=2)
        #print ('clippedfiveby3dmean',clippedfiveby3dmean)
        clippedfiveby3dmedian=np.ma.median(clippedfiveby3d,axis=2)
        #print ('clippedfiveby3dmedian',clippedfiveby3dmedian)
        header['NAXIS'] = 2

        border=np.concatenate((np.ravel(clippedfiveby3dmedian[0,:]),np.ravel(clippedfiveby3dmedian[-1,:]),np.ravel(clippedfiveby3dmedian[1:6,0]),np.ravel(clippedfiveby3dmedian[1:6,-1])))
        normclippedfiveby3dmedian=clippedfiveby3dmedian.data-np.mean(border)
        normclippedfiveby3dmedian=normclippedfiveby3dmedian/np.sum(normclippedfiveby3dmedian)
        
        #trim & enforce symmetry for some pixels
        normclippedfiveby3dmedian=normclippedfiveby3dmedian[1:6,1:6]
        mask1=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
        mask1[2,1]=1
        mask1[2,3]=1
        mask2=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
        mask2[1,1]=1
        mask2[1,3]=1
        mask2[3,1]=1
        mask2[3,3]=1
        mask3=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
        mask3[0,2]=1
        mask3[4,2]=1
        mask3[2,0]=1
        mask3[2,4]=1
        mask4=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
        mask4[0,1]=1
        mask4[0,3]=1
        mask4[1,0]=1
        mask4[1,4]=1
        mask4[3,0]=1
        mask4[3,4]=1
        mask4[4,1]=1
        mask4[4,3]=1
        mask5=np.zeros((normclippedfiveby3dmedian.shape),'uint16')
        mask5[0,0]=1
        mask5[0,4]=1
        mask5[4,0]=1
        mask5[4,4]=1

        #set option of enforced symmetry
        if makesymm==True:
            w=np.where(mask1==1)
            normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
            w=np.where(mask2==1)
            normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
            w=np.where(mask3==1)
            normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
            w=np.where(mask4==1)
            normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])
            w=np.where(mask5==1)
            normclippedfiveby3dmedian[w]=np.mean(normclippedfiveby3dmedian[w])

        normclippedfiveby3dmedianinvoid=deepcopy(normclippedfiveby3dmedian)

        filemedian='ipc5by5median_amp%s_invoid.fits' %  (amplifier[j])
        fits.writeto(filemedian,normclippedfiveby3dmedian,header,overwrite=True)

        print (normclippedfiveby3dmedian[2,1])
        print (fiveby3d.shape,np.mean(border))

        #expand shape of array
        normclippedfiveby3dmedianinvoidexpand=np.expand_dims(normclippedfiveby3dmedianinvoid,axis=2)
        normclippedfiveby3dmedianinvoidexpand=np.expand_dims(normclippedfiveby3dmedianinvoidexpand,axis=3)
        normclippedfiveby3dmedianinvoidexpand=np.repeat(normclippedfiveby3dmedianinvoidexpand,2048,axis=2)
        normclippedfiveby3dmedianinvoidexpand=np.repeat(normclippedfiveby3dmedianinvoidexpand,2048,axis=3)
        
    if j==3:
        normclippedfiveby3dmedianinvoid=np.flip(normclippedfiveby3dmedianinvoid,axis=0)
        filemedian='ipc5by5median_amp%s_invoid_symm.fits' %  (amplifier[j])
        fits.writeto(filemedian,normclippedfiveby3dmedianinvoid,header,overwrite=True)
        #expand shape of array
        normclippedfiveby3dmedianinvoidexpand=np.expand_dims(normclippedfiveby3dmedianinvoid,axis=2)
        normclippedfiveby3dmedianinvoidexpand=np.expand_dims(normclippedfiveby3dmedianinvoidexpand,axis=3)
        normclippedfiveby3dmedianinvoidexpand=np.repeat(normclippedfiveby3dmedianinvoidexpand,2048,axis=2)
        normclippedfiveby3dmedianinvoidexpand=np.repeat(normclippedfiveby3dmedianinvoidexpand,2048,axis=3)

    if j>0 and j<4:
        #put section in 4D IPC file
        w=np.where((voidmask==1)&(ampmask==1))
        print (ipc4d[:,:,w[0],w[1]].shape,normclippedfiveby3dmedianinvoidexpand[:,:,w[0],w[1]].shape)
        ipc4d[:,:,w[0],w[1]]=normclippedfiveby3dmedianinvoidexpand[:,:,w[0],w[1]]


#set all reference pixels to 1 in centre and zero elsewhere
w=np.where(refpixflagged==1)
print (refpixflagged[w].size)
ipc4d[:,:,w[0],w[1]]=0
ipc4d[2,2,w[0],w[1]]=1
     
#output 4D IPC convolution reference file
header['NAXIS'] = 4
fits.writeto(ipcfile,ipc4d,header,overwrite=True)

