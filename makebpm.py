#!/usr/bin/env python

#This routine generates three bad pixel mask files for NIRISS. The three masks have 
#different dark noise thresholds appropriate to various types of calibration or science data.
#The routine is robust to high rates of cosmic rays and separates cosmic ray hits from noisy pixels
#Inputs:
#    Darks: 2D slope count rate images including cosmic rays run through calwebb_detector1 with steps: 
#           dq_init (reference pixels only), saturation, superbias, refpix, linearity, column jump (NIRISS team specific step), ramp_fit
#    Darks: 4D cube images run through calwebb_detector1 with steps: 
#           dq_init (reference pixels only), saturation, superbias, refpix, linearity, column jump (NIRISS team specific step), jump (variant including neighbour detection at lower threshold)
#    Superbias: Superbias reference file
#    Superbias: Superbias first frame standard deviation file (sigma-clipped)
#    Flat-field: A processed NIRISS grism flat-field count rate image that does not contain pixels that are bad because of low illumination and defects
#Optional inputs (if not provided, can be generated):
#    Dark noise: Slope noise file generated from initial 2D slope count rate images including cosmic rays
#    Dark noise: CDS noise file generated from initial 4D cube images
#Outputs:
#    Bad pixel masks: Three bad pixel mask reference files with different dark noise thresholds.
#Optional outputs:
#    Darks: 4D cube images as input but with updated GROUPDQ flags to better identify only cosmic rays and not noisy pixels  
#    Dark noise: Slope noise file generated from 2D slope count rate images excluding pixels impacted by cosmic rays
#    Dark noise: CDS noise file generated from 4D cube images excluding pixels impacted by cosmic rays
#    Bad pixel masks: Separate files for each type of bad pixel identified
  

import numpy as np
from copy import deepcopy
import re
import os
import scipy.optimize as op
from astropy.io import fits
from photutils import CircularAnnulus
from photutils import aperture_photometry
import stsci.imagestats as imagestats
import natsort
from copy import deepcopy
import math
import datetime
from jwst.datamodels import dqflags

#Likelihood subroutines used in checking for RTS behaviour

#Three Gaussians fit
def lnlike(theta, bincen, histcds, histcdserr):
    a1, a2, a3, a4, a5, lnf = theta
    model=a3*np.exp((-1.0*(bincen-a1)**2.0)/(2.0*a5**2.0))+a4*np.exp((-1.0*(bincen-(a1-a2))**2.0)/(2.0*a5**2.0))+a4*np.exp((-1.0*(bincen-(a1+a2))**2.0)/(2.0*a5**2.0))
    y=histcds
    yerr=histcdserr
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#Single Gaussian fit
def lnlikesingle(theta, bincen, histcds, histcdserr):
    a1, a3, a5, lnf = theta
    model=a3*np.exp((-1.0*(bincen-a1)**2.0)/(2.0*a5**2.0))
    y=histcds
    yerr=histcdserr
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#Set up primary header and DQ_DEF tables of output files
def setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise_here,thresh_cdsnoise_here):
    #Define primary HDU with standard header
    maskhduprimary = fits.PrimaryHDU()
    prihdr = maskhduprimary.header
    prihdr['DATE']    = (datetime.datetime.utcnow().isoformat() , 'Date this file was created (UTC)')
    prihdr['FILENAME']= ('niriss_ref_bad_pixel_mask.fits' , 'Name of reference file.')
    prihdr['DATAMODL']= ('MaskModel'          , 'Type of data model')
    prihdr['TELESCOP']= ('JWST    '           , 'Telescope used to acquire the data')
    prihdr['INSTRUME']= ('NIRISS  '           , 'Instrument used to acquire the data')
    prihdr['DETECTOR']= ('NIS     '           , 'Name of detector used to acquire the data')
    prihdr['SUBARRAY']= ('GENERIC '           , 'Subarray of reference file.')
    prihdr['SUBSTRT1']= (                   1 , 'Starting pixel in axis 1 direction')
    prihdr['SUBSTRT2']= (                   1 , 'Starting pixel in axis 2 direction')
    prihdr['SUBSIZE1']= (                2048 , 'Number of pixels in axis 1 direction')
    prihdr['SUBSIZE2']= (                2048 , 'Number of pixels in axis 2 direction')
    prihdr['FASTAXIS']= (                  -2 , 'Fast readout axis direction')
    prihdr['SLOWAXIS']= (                  -1 , 'Slow readout axis direction')
    prihdr['COL_STRT']= (                   1 , 'X-position of corner (Column)')
    prihdr['ROW_STRT']= (                   1 , 'Y-position of corher (Row)')
    prihdr['REFTYPE'] = ('MASK    '           , 'Type of reference file.')
    prihdr['AUTHOR']  = ('Chris Willott')
    prihdr['PEDIGREE']= ('GROUND  ')
    prihdr['USEAFTER']= ('2015-11-01T00:00:00')
    prihdr['DESCRIP'] = ('This is a bad pixel mask reference file.---------------------------')
    prihdr['TEMP']    = (                37.7 , ' Temperature')
    prihdr.insert('TELESCOP',(''   , '' ),after=True)
    prihdr.insert('TELESCOP',(''   , 'Instrument configuration information' ),after=True)
    prihdr.insert('TELESCOP',(''   , '' ),after=True)
    prihdr['HISTORY'] = 'This file is the bad pixel reference file. etc. etc.'

    #Define DQ_DEF binary table HDU
    flagtable=np.rec.array([
               ( 0,        1, 'DO_NOT_USE',      'Bad pixel not to be used for science or calibration'   ),
               ( 1,        2, 'DEAD',            'QE < {:6.3f} and QE of the closest 4 pixels < {:6.3f}. Not RC'.format(thresh_dead,thresh_adjopen) ),
               ( 2,        4, 'HOT',             'DC > {:6.3f} e-/s'.format(thresh_hot)   ),
               ( 3,        8, 'WARM',            'DC {:6.3f} to {:6.3f} e-/s; not marked as DO_NOT_USE because warm'.format(thresh_warm,thresh_hot)   ),
               ( 4,       16, 'LOW_QE',          'QE {:6.3f} to {:6.3f} and QE of the closest 4 pixels < {:6.3f}. Not RC'.format(thresh_dead,thresh_lowqe,thresh_adjopen)  ),
               ( 5,       32, 'RC',              'Shows RC behaviour in darks'   ),
               ( 6,       64, 'TELEGRAPH',       'Shows RTS behaviour in darks; not marked as DO_NOT_USE because RTS'   ),
               ( 7,      128, 'NONLINEAR',       'From linearity reference file?'   ),  
               ( 8,      256, 'BAD_REF_PIXEL',   'All bad reference pixels'   ),
               ( 9,      512, 'UNRELIABLE_BIAS', 'Bias level > {:8.1f} and/or bias noise > {:6.3f} sigma'.format(thresh_bias,thresh_bias_sigma)   ),		
               ( 10,    1024, 'UNRELIABLE_DARK', 'CDS noise > {:6.3f} times median in long darks'.format(thresh_cdsnoise_here)   ),			
               ( 11,    2048, 'UNRELIABLE_SLOPE','Slope noise > {:6.3f} times median in long darks'.format(thresh_slopenoise_here)   ),
               ( 12,    4096, 'UNRELIABLE_FLAT', 'High noise in illuminated flat-field data'   ), 
               ( 13,    8192, 'OPEN',            'QE < {:6.3f} and QE of the closest 4 pixels > {:6.3f}. Not RC'.format(thresh_lowqe,thresh_adjopen)  ),	
               ( 14,   16384, 'ADJ_OPEN',        'One of 8 pixels near open/RC with closest 4 pixels > {:6.3f} or one of 4 pixels adjacent to open/RC with closest 4 pixels {:6.3f} to {:6.3f}'.format(thresh_nearopen,thresh_adjopen,thresh_nearopen)   ), 
               ( 15,   32768, 'OTHER_BAD_PIXEL', 'Other bad pixel type'   ),    
               ( 16,   65536, 'REFERENCE_PIXEL', 'All reference pixels'   )],
               formats='int32,int32,a40,a140',
               names='Bit,Value,Name,Description')

    maskhdudqdef = fits.BinTableHDU(flagtable,name='DQ_DEF  ',ver=1)
    return maskhduprimary,maskhdudqdef,flagtable


#NIRISS Detector 
#noise files are in units of ADU so need a value of gain to convert to electrons
gain=1.62

#Set threshold parameters that determine when a pixel is bad and the category
#Usable QE threshold 
thresh_lowqe=0.6
#QE below which to flag as DEAD
thresh_dead=0.05
#QE (normalized locally) of nearest 4 pixels to an open or RC pixel to flag as ADJ_OPEN
thresh_adjopen=1.05
#QE (normalized locally) of nearest 4 pixels to an open or RC pixel to flag further 4 pixels as ADJ_OPEN
thresh_nearopen=1.10
#Dark current above which to flag as HOT
thresh_hot=0.5
#Dark current above which to flag as WARM (not marked as DO_NOT_USE)
thresh_warm=0.1
#Dark current below which pixels are unusual and to flag as OTHER_BAD_PIXEL
thresh_cold=0.004
#Bias level above which to flag as UNRELIABLE_BIAS
thresh_bias=30000.0
#Variability in bias above which to flag as UNRELIABLE_BIAS
thresh_bias_sigma=2.0
#Dark slope noise sigma above which to always flag as UNRELIABLE_SLOPE
thresh_slopenoise=3.0
#Dark CDS noise sigma above which to always flag as UNRELIABLE_DARK
thresh_cdsnoise=3.0
#Dark slope noise sigma above which to flag as UNRELIABLE_SLOPE for long exposures only
thresh_slopenoise_longonly=2.0
#Dark CDS noise sigma above which to flag as UNRELIABLE_DARK for short exposures only
thresh_cdsnoise_shortonly=1.5


#Define reference file locations and output file locations and names
#refdir='/Users/willottc/niriss/detectors/willott_reference_files/'
refdir='/mnt/jwstdata/willott_reference_files/'
rootbpmfile=refdir+'first3runscv3/jwst_niriss_cv3_38k_bpm'
#rootbpmfile=refdir+'simcrs/jwst_niriss_cv3_38k_simcrs_bpm'
#rootbpmfile=refdir+'nocrs/jwst_niriss_cv3_38k_nocrs_bpm'
longbpmfile=rootbpmfile+'_long.fits'
shortbpmfile=rootbpmfile+'_short.fits'
minimalbpmfile=rootbpmfile+'_minimal.fits'
typebpmdir=refdir+'first3runscv3/badpixeltypes/'
#typebpmdir=refdir+'nocrs/badpixeltypes/'
typebpmfile=typebpmdir+'jwst_niriss_cv3_38k_bpm_type'

#location of dark noise, slope and CDS files
datadir='/mnt/jwstdata/cv3/colddarks/first3runs/'
darknoisedir=datadir+'noise/'
slopedir=datadir+'slope/'
cubedir=datadir+'cube/'
outcubedir=datadir+'cubefixcr/'
if not os.path.exists(darknoisedir):
    os.makedirs(darknoisedir)
if not os.path.exists(outcubedir):
    os.makedirs(outcubedir)

#If it already exists, load initial run of dark noise data (before fully corrected for cosmic rays)
#If it doesn't then make it on the fly
#multiply by gain to get in e-
#slopefileroot='cv3_38k_third2runs_init_sigmadarkzero.fits'
slopefileroot='cv3_38k_first3runs_init_sigmadarkzero.fits'
slopenoisefile=darknoisedir+slopefileroot
cdsnoisefile=slopenoisefile.replace('sigmadarkzero.fits','mediancds1fstd.fits')
noiserootout=slopefileroot.replace('sigmadarkzero.fits','')

if os.path.exists(slopenoisefile) and os.path.exists(cdsnoisefile):
    print ('using existing files',slopenoisefile,cdsnoisefile)
else:    
    print ('making initial dark current, slope and CDS noise files')
    makenoisescmd='./makedarknoisefilesgdq.py --cubedir={} --slopedir={} --noisedir={} --outfileroot={} --usegdq=False'.format(cubedir,slopedir,darknoisedir,noiserootout)
    os.system(makenoisescmd)
 
hdulist=fits.open(slopenoisefile)
header=hdulist[0].header
slopenoise=hdulist[0].data*gain
hdulist=fits.open(cdsnoisefile)
cdsnoise=hdulist[0].data*gain

#create empty arrays for all masks - some of these separate for short, long and minimal bpms
arrshape=cdsnoise.shape
bpm=np.zeros(arrshape, dtype='int32')
donotuseshort=np.zeros(arrshape, dtype='uint8')
donotuselong=np.zeros(arrshape, dtype='uint8')
donotuseminimal=np.zeros(arrshape, dtype='uint8')
dead=np.zeros(arrshape, dtype='uint8')
hot=np.zeros(arrshape, dtype='uint8')
warm=np.zeros(arrshape, dtype='uint8')
lowqe=np.zeros(arrshape, dtype='uint8')
rc=np.zeros(arrshape, dtype='uint8')
telegraph=np.zeros(arrshape, dtype='uint8')
nonlinear=np.zeros(arrshape, dtype='uint8')
badrefpixelshort=np.zeros(arrshape, dtype='uint8')
badrefpixellong=np.zeros(arrshape, dtype='uint8')
badrefpixelminimal=np.zeros(arrshape, dtype='uint8')
unrelbias=np.zeros(arrshape, dtype='uint8')
unreldarkshort=np.zeros(arrshape, dtype='uint8')
unreldarklong=np.zeros(arrshape, dtype='uint8')
unreldarkminimal=np.zeros(arrshape, dtype='uint8')
unrelslopeshort=np.zeros(arrshape, dtype='uint8')
unrelslopelong=np.zeros(arrshape, dtype='uint8')
unrelslopeminimal=np.zeros(arrshape, dtype='uint8')
unrelflat=np.zeros(arrshape, dtype='uint8')
openpixel=np.zeros(arrshape, dtype='uint8')
adjopen=np.zeros(arrshape, dtype='uint8')
otherbadshort=np.zeros(arrshape, dtype='uint8')
otherbadlong=np.zeros(arrshape, dtype='uint8')
otherbadminimal=np.zeros(arrshape, dtype='uint8')
refpixel=np.zeros(arrshape, dtype='uint8')
noisyindarks=np.zeros(arrshape, dtype='uint8')

#Identify reference pixels
refpixel+=1
refpixel[4:2044,4:2044]=0

##Dark images section##
##Get statistics separately for reference and active pixels
a=np.where(refpixel==0)
slopenoiseactive=slopenoise[a]
cdsnoiseactive=cdsnoise[a]
r=np.where(refpixel==1)
slopenoiserefpix=slopenoise[r]
cdsnoiserefpix=cdsnoise[r]

imstatslopenoiseactive = imagestats.ImageStats(slopenoiseactive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatslopenoiseactive.printStats()

imstatcdsnoiseactive = imagestats.ImageStats(cdsnoiseactive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatcdsnoiseactive.printStats()

imstatslopenoiserefpix = imagestats.ImageStats(slopenoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatslopenoiserefpix.printStats()

imstatcdsnoiserefpix = imagestats.ImageStats(cdsnoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatcdsnoiserefpix.printStats()

#set threshold used for finding RC and inverse RC pixels
#set excess at first 3 CDS points in the ramp at 7 times median
first3excesslimit=7.0*imstatcdsnoiseactive.median 
print ('first3excesslimit',first3excesslimit)

#Identify pixels noisy in darks that require typing
w=np.where(((cdsnoise>=(1.5*imstatcdsnoiseactive.median))&(refpixel==0))|((cdsnoise>=(1.5*imstatcdsnoiserefpix.median))&(refpixel==1))|((slopenoise>=(1.5*imstatslopenoiseactive.median))&(refpixel==0))|((slopenoise>=(1.5*imstatslopenoiserefpix.median))&(refpixel==1)))
noisyindarks[w]=1

#Use CDS files to get time ordered sequences of pixels
dirlist=natsort.natsorted(os.listdir(cubedir))
dirlist[:] = (value for value in dirlist if value.startswith('NISNIR') and value.endswith('.fits'))

numimages=len(dirlist)
print (numimages)

#will read in strips of data to avoid having all images in memory at once
dxstrip=100
numsec=int(2048/dxstrip+1)
numpixsec=2048*numsec
maxx=int(np.min([(numsec*dxstrip,2048)]))

#for testing
#numimages=5

for i in range(numsec):
#for testing
#for i in range(4,5):
    xmin=i*dxstrip
    xmax=min([2048,xmin+dxstrip])
    print (i,xmin,xmax)

    print ('loading cube section',datetime.datetime.now())
    for j in range(numimages):

        cubefile=cubedir+dirlist[j]
        with fits.open(cubefile,memmap=True) as hdul:
            header=hdul[0].header
            ngroup=header['NGROUPS']
            cds = np.diff(np.squeeze(hdul['SCI'].data[:,:,:,xmin:xmax]),axis=0)*gain
        cdsinteger=np.rint(cds)
        cds=cdsinteger.astype(int)
        if j==0:
            cdstime=cds
        else:
            cdstime=np.vstack((cdstime,cds))
        del cdsinteger,cds

    print (datetime.datetime.now())
    #get same section of bpm and make crhits mask
    noisyindarkssection=noisyindarks[:,xmin:xmax]
    cdsnoisesection=cdsnoise[:,xmin:xmax]
    slopenoisesection=slopenoise[:,xmin:xmax]    
    numsigcds=cdsnoisesection/imstatcdsnoiseactive.median
    crhitssection=np.zeros(cdstime.shape, dtype='uint8')
    noncrhitssection=np.zeros(cdstime.shape, dtype='uint8')

    #Make this first 3 excess using median per frame for all ramps
    xesgroup1list2d=[(np.array([0])+p*(ngroup-1)) for p in range(numimages)]
    xesgroup2list2d=[(np.array([1])+p*(ngroup-1)) for p in range(numimages)]
    xesgroup3list2d=[(np.array([2])+p*(ngroup-1)) for p in range(numimages)]
    xeslaterlist2d=[(np.arange((ngroup-30),(ngroup-3))+p*(ngroup-1)) for p in range(numimages)]
    xesgroup1list=np.ravel(xesgroup1list2d)
    xesgroup2list=np.ravel(xesgroup2list2d)
    xesgroup3list=np.ravel(xesgroup3list2d)
    numxesgroup1list=xesgroup1list.size
    xeslaterlist=np.ravel(xeslaterlist2d)
    xesgroup1median=(np.median(cdstime[xesgroup1list,:,:],axis=0))
    xesgroup2median=(np.median(cdstime[xesgroup2list,:,:],axis=0))
    xesgroup3median=(np.median(cdstime[xesgroup3list,:,:],axis=0))
    first3excess=(xesgroup1median+xesgroup2median+xesgroup3median)-3.0*(np.median(cdstime[xeslaterlist,:,:],axis=0))
    first3excess1d=np.ravel(first3excess) 

    #Select RC pixels 
    w=np.where(first3excess > first3excesslimit)
    xfull=xmin+w[1]
    rc[w[0],xfull]=1
    numrc=first3excess[w].size

    #Select 'Inverse RC' pixels and flag as "OTHER_BAD"  
    w=np.where(first3excess < (-1.0*first3excesslimit))
    xfull=xmin+w[1]
    otherbadshort[w[0],xfull]=1
    otherbadlong[w[0],xfull]=1
    otherbadminimal[w[0],xfull]=1
    numinvrc=first3excess[w].size

    #Select possible Random Telegraph Signal (RTS) pixels to be checked
    w=np.where((noisyindarkssection==1)&(first3excess <= first3excesslimit)&(first3excess >= (-1.0*first3excesslimit)))
    w0=w[0]
    w1=w[1]
    numpossiblerts=w0.size
    print (numrc,numinvrc,numpossiblerts,noisyindarkssection[(np.where(noisyindarkssection==1))].size)

    for x in range(numpossiblerts):
        #Remove most cosmic rays by iteratively clipping positive CDS values when no negative values within 80%. 
        #Subtract median to avoid finding lots of 'cosmic rays' in hot pixels
        xfull=xmin+w1[x]
        yfull=w0[x]

        cdstimesortindex=np.argsort(cdstime[:,w0[x],w1[x]])
        cdstimesorted=cdstime[cdstimesortindex,w0[x],w1[x]]-np.median(cdstime[:,w0[x],w1[x]])
        while (cdstimesorted[0]> (-0.8*cdstimesorted[-1])):
            crhitssection[cdstimesortindex[-1],w0[x],w1[x]]=1
            cdstimesorted=cdstimesorted[:-1]
            cdstimesortindex=cdstimesortindex[:-1]

        #Set up CDS bins, including +/-20 beyond last point.
        binmax=2*round((np.amax(cdstimesorted)+1)/2)+21.5
        binmin=2*round((np.amin(cdstimesorted)-1)/2)-20.5
        numbins=int((binmax-binmin+1)/2.0)
        binedges=(np.arange(numbins+1)*(binmax-binmin)/numbins)+binmin
        bincen=binedges[:-1]+1.0
        histcds,edges = np.histogram(cdstimesorted,binedges)
        histcdserr=histcds**0.5
        histcdserr[np.where(histcds==0)]=1.0

        #Fit three constrained Gaussians to the CDS histogram
        #a1 - peak of central Gaussian, should be close to zero for darks
        a1init=0.0
        #a2 - offset from a1 of other two peaks, initially set to 35% of max-min value
        a2init=0.35*(binmax-binmin)
        #a3 - height of central Gaussian, initially set to max of histogram
        a3init=np.amax(histcds)
        #a4 - height of other 2 Gaussians, variable depending upon RTS frequency
        a4init=0.3*a3init
        #a5 - dispersion of all 3 Gaussians - initially set to median CDS noise
        a5init=imstatcdsnoiseactive.median
        lnfinit=-3.0
        paramsinit=[a1init,a2init,a3init,a4init,a5init,lnfinit]

        # Find the maximum likelihood value. 
        #set bounds on a4init so secondary peak at least 2% of main peak.
        #set bounds on a5init so sigma of gaussians similar to CDS noise
        chi2 = lambda *args: -2 * lnlike(*args)
        result = op.minimize(chi2, paramsinit, args=(bincen, histcds, histcdserr), method='SLSQP', bounds=((None, None), (15.0, None), (0.0, None), (0.02*a3init, None), (0.7*imstatcdsnoiseactive.median, 1.8*imstatcdsnoiseactive.median), (None, None)))
        a1_ml, a2_ml, a3_ml, a4_ml, a5_ml, lnf_ml = result["x"]

        maxlichisq=result["fun"]
        maxliresult=result["x"]
        maxliresult[4]=np.absolute(maxliresult[4])

        bestmodel=a3_ml*np.exp((-1.0*(bincen-a1_ml)**2.0)/(2.0*a5_ml**2.0))+a4_ml*np.exp((-1.0*(bincen-(a1_ml-a2_ml))**2.0)/(2.0*a5_ml**2.0))+a4_ml*np.exp((-1.0*(bincen-(a1_ml+a2_ml))**2.0)/(2.0*a5_ml**2.0))

        #Fit single Gaussian for comparison using measured CDS noise for this pixel
        a1singleinit=0.0
        a3singleinit=np.amax(histcds)
        a5singleinit=cdsnoisesection[w0[x],w1[x]]
        lnfsingleinit=-3.0
        singleparamsinit=[a1singleinit,a3singleinit,a5singleinit,lnfsingleinit]

        # Find the maximum likelihood value.
        chi2single = lambda *args: -2 * lnlikesingle(*args)
        resultsingle  = op.minimize(chi2single, singleparamsinit, args=(bincen, histcds, histcdserr), method='SLSQP', bounds=((None, None), (0.0, None), (0.7*imstatcdsnoiseactive.median, None), (None, None)))
        a1single_ml, a3single_ml, a5single_ml, lnf_ml = resultsingle["x"]
        maxlichisqsingle=resultsingle["fun"]
        maxliresultsingle=resultsingle["x"]
        bestmodelsingle=a3single_ml*np.exp((-1.0*(bincen-a1single_ml)**2.0)/(2.0*a5single_ml**2.0))

        #check if BIC shows extra parameters are justified
        #BIC=-2 ln L + k*(ln n - ln 2pi) where L lis likeli, k is num param, n is numbins
        bicthreehist=maxlichisq+5.0*(math.log(numbins)-math.log(6.28))
        bicsinglehist=maxlichisqsingle+3.0*(math.log(numbins)-math.log(6.28))
        bicdiff=(bicsinglehist-bicthreehist)

        a3mlfrac=a3_ml/(a3_ml+2.0*a4_ml)
        a4mlfrac=1.0-a3mlfrac 

        #If BIC difference > 20 and fit is well-constrained, flag as telegraph
        if ((a4mlfrac>0.005)&(a4mlfrac<0.99)&(bicdiff>20)):
            telegraph[yfull,xfull]=1

    del cdstime

    #Set up arrays for locations of crhits and non-crhits in noisy pixels 
    if 'crhits' not in locals():
        crhits=np.zeros((((ngroup-1)*numimages),2048,2048), dtype='bool')
        noncrhits=np.zeros((((ngroup-1)*numimages),2048,2048), dtype='bool')
    crhits[:,:,xmin:xmax]=crhitssection
    #First set all noisy pixels in noncrhits to true 
    noncrhitssection[:,w0,w1]=True
    noncrhits[:,:,xmin:xmax]=noncrhitssection
    
    del noisyindarkssection,cdsnoisesection,slopenoisesection,numsigcds,crhitssection,noncrhitssection
 
#Unset all pixels containing cosmic rays from noncrhits
noncrhits[np.where(crhits==True)]=False

#Open dark cubes and for pixels that were checked because possible RTS noise edit their GROUPDQ to:
#1. remove jump flags that are probably not cosmic rays
#2. add jump flags when detected by this routine in case they were not found by jumpstep
dirlist=natsort.natsorted(os.listdir(cubedir))
dirlist[:] = (value for value in dirlist if value.startswith('NISNIR') and value.endswith('.fits'))

#set to false if this step has already been run or using files without cosmic rays
editcubegroupdq=True

if editcubegroupdq==True:

    for j in range(numimages):

        cubefile=cubedir+dirlist[j]
        outcubefile=outcubedir+dirlist[j]
        outcubefile=outcubefile.replace('.fits','_fixcr.fits')
        print (j,dirlist[j],outcubefile)

        crhitsthisimage=crhits[j*(ngroup-1):(j+1)*(ngroup-1),:,:]
        crhitsthisimage=np.pad(crhitsthisimage,((1,0),(0,0),(0,0)), 'minimum')
        noncrhitsthisimage=noncrhits[j*(ngroup-1):(j+1)*(ngroup-1),:,:]
        noncrhitsthisimage=np.pad(noncrhitsthisimage,((1,0),(0,0),(0,0)), 'maximum')

        hdul=fits.open(cubefile)
        gdq=hdul['GROUPDQ'].data

        w=np.where(gdq[0,:,:,:]==dqflags.group['JUMP_DET'])
        print ('total of %s jump flags before changes' %  (gdq[0,w[0],w[1],w[2]].size))

        #remove jump flags that are probably not cosmic rays
        w=np.where((noncrhitsthisimage==True)&(gdq[0,:,:,:]==dqflags.group['JUMP_DET']))
        gdq[0,w[0],w[1],w[2]]  = gdq[0,w[0],w[1],w[2]] -  dqflags.group['JUMP_DET']
        print ('removed %s jump flags' %  (gdq[0,w[0],w[1],w[2]].size))

        #add jump flags when detected by this routine in case they were not found by jumpstep
        w=np.where(crhitsthisimage==True)
        gdq[0,w[0],w[1],w[2]]  = np.bitwise_or(gdq[0,w[0],w[1],w[2]], dqflags.group['JUMP_DET'])
        print ('added %s jump flags' %  (gdq[0,w[0],w[1],w[2]].size))

        #add in neighbours of all large jumps (>3000) in case they were removed
        cds=np.diff(np.squeeze(hdul['SCI'].data),axis=0)
        w=np.where((cds>3000)&(gdq[0,1:,:,:]==dqflags.group['JUMP_DET']))
        gdq[0,w[0]+1,w[1]+1,w[2]]  = np.bitwise_or(gdq[0,w[0]+1,w[1]+1,w[2]], dqflags.group['JUMP_DET'])
        gdq[0,w[0]+1,w[1]-1,w[2]]  = np.bitwise_or(gdq[0,w[0]+1,w[1]-1,w[2]], dqflags.group['JUMP_DET'])
        gdq[0,w[0]+1,w[1],w[2]+1]  = np.bitwise_or(gdq[0,w[0]+1,w[1],w[2]+1], dqflags.group['JUMP_DET'])
        gdq[0,w[0]+1,w[1],w[2]-1]  = np.bitwise_or(gdq[0,w[0]+1,w[1],w[2]-1], dqflags.group['JUMP_DET'])

        hdul['GROUPDQ'].data=gdq
        hdul.writeto(outcubefile,overwrite=True)

#Run makedarknoisefilesgdq.py again now that GROUPDQ have been updated
#Set to false if this step has already been run or using files without cosmic rays where GROUPDQ was not updated
#Note that file *_meancdsstd.fits is equivalent to the pipeline readnoise reference file, except we use GDQ flagging to exclude cosmic rays
remakenoisefiles=True

if remakenoisefiles==True:
    noiserootout=noiserootout.replace('_init_','_gdq_')
    slopenoisefile=slopenoisefile.replace('_init_','_gdq_')
    cdsnoisefile=cdsnoisefile.replace('_init_','_gdq_')
    makenoisescmd='./makedarknoisefilesgdq.py --cubedir={} --slopedir={} --noisedir={} --outfileroot={} --usegdq=True'.format(outcubedir,slopedir,darknoisedir,noiserootout)
    os.system(makenoisescmd)
 
#Load final run of dark noise data (after fully corrected for cosmic rays)
#multiply by gain to get in e-
hdulist=fits.open(slopenoisefile)
header=hdulist[0].header
slopenoise=hdulist[0].data*gain

darkcurrfile=slopenoisefile.replace('_sigmadarkzero.fits','_mediandark.fits')
hdulist=fits.open(darkcurrfile)
darkcurr=hdulist[0].data*gain

hdulist=fits.open(cdsnoisefile)
cdsnoise=hdulist[0].data*gain

#Load superbias file
superbiasfile=refdir+'NIRISS_superbias_cv3_38k_dms.fits'
hdulist=fits.open(superbiasfile)
header=hdulist[0].header
superbias=hdulist[0].data

#Load superbias sigma file
superbiassigmafile=refdir+'superbiasfirstframeclippedstddev_cv3_38k_144_dms.fits'
hdulist=fits.open(superbiassigmafile)
header=hdulist[0].header
superbiassigma=hdulist[0].data

#Get normalized illuminated flat-field data (from CV3)
#Use F115W grism flat that is the imaging flat with POM defect correction applied and coronagraphic spots patched.
#will also use CDS and/or RC to identify open pixels and differentiate open from RC 
flat1file=refdir+'jwst_niriss_cv3_grismflat_F115W.fits'
#flat1file=flatsdir+'NIST74-F115-FL-6003020347_1_496_SE_2016-01-03T02h19m52_slope_norm.fits'
#flat2file=flatsdir+'NIST74-F200-FL-6003024159_1_496_SE_2016-01-03T02h55m43_slope_norm.fits'

hdulist=fits.open(flat1file)
flat1=hdulist['SCI'].data

#Recalculate noises here...
##Dark images section##
##Get statistics separately for reference and active pixels
a=np.where(refpixel==0)
slopenoiseactive=slopenoise[a]
cdsnoiseactive=cdsnoise[a]
darkcurractive=darkcurr[a]
r=np.where(refpixel==1)
slopenoiserefpix=slopenoise[r]
cdsnoiserefpix=cdsnoise[r]

imstatslopenoiseactive = imagestats.ImageStats(slopenoiseactive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatslopenoiseactive.printStats()

imstatcdsnoiseactive = imagestats.ImageStats(cdsnoiseactive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatcdsnoiseactive.printStats()

imstatdarkcurractive = imagestats.ImageStats(darkcurractive,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatdarkcurractive.printStats()

imstatslopenoiserefpix = imagestats.ImageStats(slopenoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatslopenoiserefpix.printStats()

imstatcdsnoiserefpix = imagestats.ImageStats(cdsnoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatcdsnoiserefpix.printStats()

#All high superbias pixels
w=np.where(superbias>thresh_bias)
unrelbias[w]=1

#All variable superbias pixels - threshold at 2xmedian.
#print (np.median(superbiassigma))
w=np.where(superbiassigma>thresh_bias_sigma*(np.median(superbiassigma)))
unrelbias[w]=1
#make sure includes full last column of reference pixels for NIRISS
unrelbias[:,2047]=1

#mask very high dark current (>0.5 e-/s, ~13x median). Below this level some are neighbours of hot pixels with well-behaved noise.
w=np.where(darkcurr>=thresh_hot)
hot[w]=1

#add warm pixels to mask but do not flag as DO_NOT_USE
w=np.where((darkcurr>=thresh_warm)&(darkcurr<thresh_hot))
warm[w]=1

#mask very low dark current (cold, <0.005 e-/s).
#Most of these are also other noisy types, but some are adjacent to hot pixels and do not show in flats or noise
w=np.where((darkcurr<=thresh_cold)&(refpixel==0))
otherbadshort[w]=1
otherbadlong[w]=1
otherbadminimal[w]=1
print (otherbadlong[w].size)

#mask very noisy in slope or cds for both bpm types
w=np.where(((slopenoise>=(thresh_slopenoise*imstatslopenoiseactive.median))&(refpixel==0))|((slopenoise>=(thresh_slopenoise*imstatslopenoiserefpix.median))&(refpixel==1)))
unrelslopelong[w]=1
unrelslopeshort[w]=1
unrelslopeminimal[w]=1
print (unrelslopelong[w].size)
w=np.where(((cdsnoise>=(thresh_cdsnoise*imstatcdsnoiseactive.median))&(refpixel==0))|((cdsnoise>=(thresh_cdsnoise*imstatcdsnoiserefpix.median))&(refpixel==1)))
unreldarkshort[w]=1
unreldarklong[w]=1
unreldarkminimal[w]=1
print (unreldarkshort[w].size)

#mask less noisy in slope only for long ramp bpm
w=np.where(((slopenoise>=(thresh_slopenoise_longonly*imstatslopenoiseactive.median))&(refpixel==0))|((slopenoise>=(thresh_slopenoise_longonly*imstatslopenoiserefpix.median))&(refpixel==1)))
unrelslopelong[w]=1
print (unrelslopelong[w].size)

#mask less noisy in cds only for short ramp bpm
w=np.where(((cdsnoise>=(thresh_cdsnoise_shortonly*imstatcdsnoiseactive.median))&(refpixel==0))|((cdsnoise>=(thresh_cdsnoise_shortonly*imstatcdsnoiserefpix.median))&(refpixel==1)))
unreldarkshort[w]=1
print (unreldarkshort[w].size)

##Illuminated images section##

#UNRELIABLE_FLAT for all pixels at active pixel border because 10% higher in flats
#sections [4:5,4:2044], [2043:2044,4:2044], [5:2043,4:5], [5:2043,2043:2044]
unrelflat[4:5,4:2044]=1
unrelflat[2043:2044,4:2044]=1
unrelflat[5:2043,4:5]=1
unrelflat[5:2043,2043:2044]=1

#make version of flat without high or low pixels to use in determining background for bad neighbours
#make unity version of flat to use as area because area function does not account for out of boundary
flat1unity=np.ones(flat1.shape)
flat1clipped=deepcopy(flat1)
v=np.where((flat1<thresh_lowqe)|(flat1>1.2))
flat1clipped[v]=1.0

#check to see if an open pixel with bad neighbours or low QE or dead
#Include all RC pixels in this check because they can also give flux to neighours
#One of 8 pixels adjacent to open pixel with closest 4 pixels > 1.1 or 4 pixels adjacent to open pixel with closest 4 pixels in range 1.05 to 1.1
#Normalize neighbours by local QE

w=np.where((flat1<=thresh_lowqe)|(rc==1))
numlo=w[0].size
ally=w[0]
allx=w[1]


for k in range(numlo):
    y=ally[k]
    x=allx[k]
    if x>4 and x<2043 and y>4 and y<2043: 
        #compare1=([y-10:y-3,x]) 

        annulus_apertures = CircularAnnulus([(x,y)], r_in=3., r_out=10.)
        phot_table1 = aperture_photometry(flat1clipped, annulus_apertures)
        phot_table2 = aperture_photometry(flat1unity, annulus_apertures)
        bkg_mean = phot_table1['aperture_sum'] / phot_table2['aperture_sum'] 

        #if adjacent pixels have low qe, set those to the background for this test
        neigh1=flat1[y+1,x]
        neigh2=flat1[y-1,x]
        neigh3=flat1[y,x+1]
        neigh4=flat1[y,x-1]
        if neigh1<thresh_lowqe:
            neigh1=bkg_mean
        if neigh2<thresh_lowqe:
            neigh2=bkg_mean
        if neigh3<thresh_lowqe:
            neigh3=bkg_mean
        if neigh4<thresh_lowqe:
            neigh4=bkg_mean
        fluxneigh1=(neigh1+neigh2+neigh3+neigh4)/4.0

        neighexcess=fluxneigh1/bkg_mean[0]
        if neighexcess>=thresh_nearopen:
            adjopen[y+1,x+1]=1
            adjopen[y-1,x-1]=1
            adjopen[y-1,x+1]=1
            adjopen[y+1,x-1]=1
        if neighexcess>=thresh_adjopen:
            openpixel[y,x]=1
            adjopen[y+1,x]=1
            adjopen[y-1,x]=1
            adjopen[y,x+1]=1
            adjopen[y,x-1]=1
        else:
            if flat1[y,x]<=thresh_dead:
                dead[y,x]=1
            else:    
                lowqe[y,x]=1

#Reset any adjacent to open that are actually RC, low qe in clusters, or unreliable bias that are also low qe
w=np.where((rc==1)|(openpixel==1)|(dead==1)|(lowqe==1))
adjopen[w]=0
unrelbias[w]=0

#Reset from these categories all RC pixels
w=np.where(rc==1)
openpixel[w]=0
dead[w]=0
lowqe[w]=0

#Reset "other bad" category if already in a different category
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(telegraph==1)|(nonlinear==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(unreldarkshort==1)|(unrelslopeshort==1))
otherbadshort[w]=0
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(telegraph==1)|(nonlinear==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(unreldarklong==1)|(unrelslopelong==1))
otherbadlong[w]=0
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(telegraph==1)|(nonlinear==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(unreldarkminimal==1)|(unrelslopeminimal==1))
otherbadminimal[w]=0

#Set DO_NOT_USE short and long for some bad pixel types
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(nonlinear==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(badrefpixelshort==1)|(unreldarkshort==1)|(unrelslopeshort==1)|(otherbadshort==1))
donotuseshort[w]=1
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(nonlinear==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(badrefpixellong==1)|(unreldarklong==1)|(unrelslopelong==1)|(otherbadlong==1))
donotuselong[w]=1
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(nonlinear==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(badrefpixelminimal==1)|(unreldarkminimal==1)|(unrelslopeminimal==1)|(otherbadminimal==1))
donotuseminimal[w]=1

#Set up primary header and DQ_DEF tables of output files
maskhduprimary,maskhdudqdeflong,flagtable=setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise_longonly,thresh_cdsnoise)

maskhduprimary,maskhdudqdefshort,flagtable=setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise,thresh_cdsnoise_shortonly)

maskhduprimary,maskhdudqdefminimal,flagtable=setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise,thresh_cdsnoise)

#Assign bad pixel bit values to bpm
bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'REFERENCE_PIXEL')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(refpixel>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'DEAD')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(dead>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'HOT')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(hot>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'WARM')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(warm>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'LOW_QE')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(lowqe>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'RC')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(rc>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'TELEGRAPH')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(telegraph>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'NONLINEAR')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(nonlinear>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_BIAS')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(unrelbias>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_FLAT')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(unrelflat>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'OPEN')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(openpixel>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'ADJ_OPEN')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(adjopen>0)
bpm[w]=np.bitwise_or(bpm[w],flagarray[w])


#Assign bad pixel bit values to bpm for attributes that are different for short, long and minimal darks

bpmlong=deepcopy(bpm)
bpmshort=deepcopy(bpm)
bpmminimal=deepcopy(bpm)

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_DARK')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(unreldarklong>0)
bpmlong[w]=np.bitwise_or(bpmlong[w],flagarray[w])
w=np.where(unreldarkshort>0)
bpmshort[w]=np.bitwise_or(bpmshort[w],flagarray[w])
w=np.where(unreldarkminimal>0)
bpmminimal[w]=np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_SLOPE')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(unrelslopelong>0)
bpmlong[w]=np.bitwise_or(bpmlong[w],flagarray[w])
w=np.where(unrelslopeshort>0)
bpmshort[w]=np.bitwise_or(bpmshort[w],flagarray[w])
w=np.where(unrelslopeminimal>0)
bpmminimal[w]=np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'DO_NOT_USE')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(donotuselong>0)
bpmlong[w]=np.bitwise_or(bpmlong[w],flagarray[w])
w=np.where(donotuseshort>0)
bpmshort[w]=np.bitwise_or(bpmshort[w],flagarray[w])
w=np.where(donotuseminimal>0)
bpmminimal[w]=np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'OTHER_BAD_PIXEL')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where(otherbadlong>0)
bpmlong[w]=np.bitwise_or(bpmlong[w],flagarray[w])
w=np.where(otherbadshort>0)
bpmshort[w]=np.bitwise_or(bpmshort[w],flagarray[w])
w=np.where(otherbadminimal>0)
bpmminimal[w]=np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'BAD_REF_PIXEL')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w=np.where((donotuselong>0)&(refpixel>0))
badrefpixellong[w]=1
bpmlong[w]=np.bitwise_or(bpmlong[w],flagarray[w])
w=np.where((donotuseshort>0)&(refpixel>0))
badrefpixelshort[w]=1
bpmshort[w]=np.bitwise_or(bpmshort[w],flagarray[w])
w=np.where((donotuseminimal>0)&(refpixel>0))
badrefpixelminimal[w]=1
bpmminimal[w]=np.bitwise_or(bpmminimal[w],flagarray[w])

maskhdudqlong=fits.ImageHDU(bpmlong,name='DQ')
maskhdudqshort=fits.ImageHDU(bpmshort,name='DQ')
maskhdudqminimal=fits.ImageHDU(bpmminimal,name='DQ')

maskhdulist=fits.HDUList([maskhduprimary,maskhdudqlong,maskhdudqdeflong])
maskhdulist.writeto(longbpmfile, overwrite=True)

maskhdulist=fits.HDUList([maskhduprimary,maskhdudqshort,maskhdudqdefshort])
maskhdulist.writeto(shortbpmfile, overwrite=True)

maskhdulist=fits.HDUList([maskhduprimary,maskhdudqminimal,maskhdudqdefminimal])
maskhdulist.writeto(minimalbpmfile, overwrite=True)


outputseparatefiles=True

if outputseparatefiles==True:
    if not os.path.exists(typebpmdir):
        os.makedirs(typebpmdir)
    donotuselongfile=typebpmfile+'_donotuse_long.fits'
    fits.writeto(donotuselongfile,donotuselong,header,overwrite=True)
    donotuseshortfile=typebpmfile+'_donotuse_short.fits'
    fits.writeto(donotuseshortfile,donotuseshort,header,overwrite=True)
    donotuseminimalfile=typebpmfile+'_donotuse_minimal.fits'
    fits.writeto(donotuseminimalfile,donotuseminimal,header,overwrite=True)
    deadfile=typebpmfile+'_dead.fits'
    fits.writeto(deadfile,dead,header,overwrite=True)
    hotfile=typebpmfile+'_hot.fits'
    fits.writeto(hotfile,hot,header,overwrite=True)
    warmfile=typebpmfile+'_warm.fits'
    fits.writeto(warmfile,warm,header,overwrite=True)
    lowqefile=typebpmfile+'_lowqe.fits'
    fits.writeto(lowqefile,lowqe,header,overwrite=True)
    rcfile=typebpmfile+'_rc.fits'
    fits.writeto(rcfile,rc,header,overwrite=True)
    telegraphfile=typebpmfile+'_telegraph.fits'
    fits.writeto(telegraphfile,telegraph,header,overwrite=True)
    nonlinearfile=typebpmfile+'_nonlinear.fits'
    fits.writeto(nonlinearfile,nonlinear,header,overwrite=True)
    badrefpixellongfile=typebpmfile+'_badrefpixel_long.fits'
    fits.writeto(badrefpixellongfile,badrefpixellong,header,overwrite=True)
    badrefpixelshortfile=typebpmfile+'_badrefpixel_short.fits'
    fits.writeto(badrefpixelshortfile,badrefpixelshort,header,overwrite=True)
    badrefpixelminimalfile=typebpmfile+'_badrefpixel_minimal.fits'
    fits.writeto(badrefpixelminimalfile,badrefpixelminimal,header,overwrite=True)
    unrelbiasfile=typebpmfile+'_unrelbias.fits'
    fits.writeto(unrelbiasfile,unrelbias,header,overwrite=True)
    unreldarklongfile=typebpmfile+'_unreldark_long.fits'
    fits.writeto(unreldarklongfile,unreldarklong,header,overwrite=True)
    unreldarkshortfile=typebpmfile+'_unreldark_short.fits'
    fits.writeto(unreldarkshortfile,unreldarkshort,header,overwrite=True)
    unreldarkminimalfile=typebpmfile+'_unreldark_minimal.fits'
    fits.writeto(unreldarkminimalfile,unreldarkminimal,header,overwrite=True)
    unrelslopelongfile=typebpmfile+'_unrelslope_long.fits'
    fits.writeto(unrelslopelongfile,unrelslopelong,header,overwrite=True)
    unrelslopeshortfile=typebpmfile+'_unrelslope_short.fits'
    fits.writeto(unrelslopeshortfile,unrelslopeshort,header,overwrite=True)
    unrelslopeminimalfile=typebpmfile+'_unrelslope_minimal.fits'
    fits.writeto(unrelslopeminimalfile,unrelslopeminimal,header,overwrite=True)
    unrelflatfile=typebpmfile+'_unrelflat.fits'
    fits.writeto(unrelflatfile,unrelflat,header,overwrite=True)
    openpixelfile=typebpmfile+'_openpixel.fits'
    fits.writeto(openpixelfile,openpixel,header,overwrite=True)
    adjopenfile=typebpmfile+'_adjopen.fits'
    fits.writeto(adjopenfile,adjopen,header,overwrite=True)
    otherbadlongfile=typebpmfile+'_otherbad_long.fits'
    fits.writeto(otherbadlongfile,otherbadlong,header,overwrite=True)
    otherbadshortfile=typebpmfile+'_otherbad_short.fits'
    fits.writeto(otherbadshortfile,otherbadshort,header,overwrite=True)
    otherbadminimalfile=typebpmfile+'_otherbad_minimal.fits'
    fits.writeto(otherbadminimalfile,otherbadminimal,header,overwrite=True)
    refpixelfile=typebpmfile+'_refpixel.fits'
    fits.writeto(refpixelfile,refpixel,header,overwrite=True)

    #output number of bad pixels per type file
    w=np.where(donotuselong==1)
    print ('donotuselong; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(donotuseshort==1)
    print ('donotuseshort; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(donotuseminimal==1)
    print ('donotuseminimal; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(dead==1)
    print ('dead; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(hot==1)
    print ('hot; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(warm==1)
    print ('warm; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(lowqe==1)
    print ('lowqe; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(rc==1)
    print ('rc; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(telegraph==1)
    print ('telegraph; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(nonlinear==1)
    print ('nonlinear; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(badrefpixellong==1)
    print ('badrefpixellong; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*8.0+2040.0*8.0)))
    w=np.where(badrefpixelshort==1)
    print ('badrefpixelshort; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*8.0+2040.0*8.0)))
    w=np.where(badrefpixelminimal==1)
    print ('badrefpixelminimal; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*8.0+2040.0*8.0)))
    w=np.where(unrelbias==1)
    print ('unrelbias; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unreldarklong==1)
    print ('unreldarklong; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unreldarkshort==1)
    print ('unreldarkshort; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unreldarkminimal==1)
    print ('unreldarkminimal; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unrelslopelong==1)
    print ('unrelslopelong; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unrelslopeshort==1)
    print ('unrelslopeshort; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unrelslopeminimal==1)
    print ('unrelslopeminimal; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(unrelflat==1)
    print ('unrelflat; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(openpixel==1)
    print ('openpixel; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(adjopen==1)
    print ('adjopen; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(otherbadlong==1)
    print ('otherbadlong; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(otherbadshort==1)
    print ('otherbadshort; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    w=np.where(otherbadminimal==1)
    print ('otherbadminimal; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))

    #w=np.where((hot==1)&(dead==1))
    #print ('hot&dead; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))
    #w=np.where((otherbadshort==1)&(unrelbias==1))
    #print ('cold&unrelbias; N=',len(w[0]),', %=',(100.0*len(w[0])/(2048.0*2048.0)))

#with cosmic rays
#donotuselong; N= 54584 , %= 1.3013839721679688
#donotuseshort; N= 100690 , %= 2.400636672973633
#donotuseminimal; N= 38202 , %= 0.9108066558837891
#dead; N= 3640 , %= 0.08678436279296875
#hot; N= 6133 , %= 0.14622211456298828
#warm; N= 9845 , %= 0.23472309112548828
#lowqe; N= 1116 , %= 0.026607513427734375
#rc; N= 5422 , %= 0.1292705535888672
#telegraph; N= 35799 , %= 0.8535146713256836
#nonlinear; N= 0 , %= 0.0
#badrefpixellong; N= 2451 , %= 7.494496086105675
#badrefpixelshort; N= 2597 , %= 7.9409246575342465
#badrefpixelminimal; N= 2185 , %= 6.681139921722114
#unrelbias; N= 3322 , %= 0.07920265197753906
#unreldarklong; N= 10413 , %= 0.24826526641845703
#unreldarkshort; N= 75753 , %= 1.8060922622680664
#unreldarkminimal; N= 10413 , %= 0.24826526641845703
#unrelslopelong; N= 30137 , %= 0.7185220718383789
#unrelslopeshort; N= 12315 , %= 0.2936124801635742
#unrelslopeminimal; N= 12315 , %= 0.2936124801635742
#unrelflat; N= 8156 , %= 0.19445419311523438
#openpixel; N= 949 , %= 0.02262592315673828
#adjopen; N= 4914 , %= 0.11715888977050781
#otherbadlong; N= 24 , %= 0.00057220458984375
#otherbadshort; N= 23 , %= 0.0005483627319335938
#otherbadminimal; N= 25 , %= 0.0005960464477539062

#without comsic rays using maxiter=3 sigma-clipping

