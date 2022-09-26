#!/usr/bin/env python
#Module to make NIRISS bad pixel mask and readnoise reference files.
#All input parameters should be listed in a yaml configuration file.
#usage ./makebpmreadnoise.py configfile
#e.g. ./makebpmreadnoise.py bpmreadnoise_nis006_20220620.cfg

#This routine generates three bad pixel masks and one readnoise reference files for NIRISS. The three masks have 
#different dark noise thresholds appropriate to various types of calibration or science data.
#The routine is robust to high rates of cosmic rays and separates cosmic ray hits from noisy pixels
#The required input files are:
#    A set of full-frame NISRAPID raw darks
#    A processed NIRISS 'grism' GR150 flat-field count rate image that does not contain pixels that are bad because of low illumination and defects
#The optional inputs are (if not provided they will be generated; all except superbias must conform to expected directory and naming structure, i.e. have been made from a previous run of this software):
#    Superbias: Superbias reference file including 'SCI' and 'ERR' extensions
#    Darks: 2D slope count rate images including cosmic rays run through calwebb_detector1 with steps: 
#           dq_init (reference pixels only), saturation, superbias, refpix, linearity, column jump (NIRISS team specific step), ramp_fit
#    Darks: 4D cube images run through calwebb_detector1 with steps: 
#           dq_init (reference pixels only), saturation, superbias, refpix, linearity, column jump (NIRISS team specific step), jump 
#    Dark noise: Slope noise file generated from initial 2D slope count rate images including cosmic rays
#    Dark noise: CDS noise file generated from initial 4D cube images
#Outputs:
#    Bad pixel masks: Three bad pixel mask reference files with different dark noise thresholds.
#    Readnoise reference file. The units are DN and it is the CDS (Correlated Double Sampling) read noise.
#Optional outputs:
#    Bad pixel masks: Separate files for each type of bad pixel identified


import numpy as np
from copy import deepcopy
import os, glob
import argparse
import scipy.optimize as op
from astropy.io import fits
from astropy.table import Table
from photutils import CircularAnnulus
from photutils import aperture_photometry
import stsci.imagestats as imagestats
from copy import deepcopy
import math
import datetime
import logging
import yaml
import natsort

from jwst.datamodels import dqflags
from makesuperbias_jwst_reffiles import run_superbias
from jwst_reffiles.dark_current import dark_reffile
#from runDetector1_niriss_fullframedark import det1_pipeline
from runDetector1_niriss_fullframedark_onefile import det1_pipeline_onefile

#==========================================================
#Functions

#Function to determine likelihood of three Gaussians fit used in checking for RTS behaviour
def lnlike(theta, bincen, histcds, histcdserr):
    a1, a2, a3, a4, a5, lnf = theta
    model = a3*np.exp((-1.0*(bincen-a1)**2.0)/(2.0*a5**2.0))+a4*np.exp((-1.0*(bincen-(a1-a2))**2.0)/(2.0*a5**2.0))+a4*np.exp((-1.0*(bincen-(a1+a2))**2.0)/(2.0*a5**2.0))
    y = histcds
    yerr = histcdserr
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#Function to determine likelihood of a single Gaussian fit used in checking for RTS behaviour
def lnlikesingle(theta, bincen, histcds, histcdserr):
    a1, a3, a5, lnf = theta
    model = a3*np.exp((-1.0*(bincen-a1)**2.0)/(2.0*a5**2.0))
    y = histcds
    yerr = histcdserr
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

#Function to set up primary header of output readnoise reference file
def setupreadnoisehdu(author,pedigree,useafter,readpattern,fpa_temp,description_readnoise,history_readnoise):
    #Define primary HDU with standard header
    readnoisehduprimary = fits.PrimaryHDU()
    readnoiseprihdr = readnoisehduprimary.header
    readnoiseprihdr['DATE']    = (datetime.datetime.utcnow().isoformat() , 'Date this file was created (UTC)')
    readnoiseprihdr['FILENAME'] =  ('jwst_niriss_readnoise.fits' , 'Name of reference file.')
    readnoiseprihdr['DATAMODL'] =  ('ReadnoiseModel'          , 'Type of data model')
    readnoiseprihdr['TELESCOP'] =  ('JWST    '           , 'Telescope used to acquire the data')
    readnoiseprihdr['INSTRUME'] =  ('NIRISS  '           , 'Instrument used to acquire the data')
    readnoiseprihdr['DETECTOR'] =  ('NIS     '           , 'Name of detector used to acquire the data')
    readnoiseprihdr['READPATT'] =  (readpattern          , 'Readout pattern')
    readnoiseprihdr['SUBARRAY'] =  ('GENERIC '           , 'Subarray of reference file.')
    readnoiseprihdr['SUBSTRT1'] =  (                   1 , 'Starting pixel in axis 1 direction')
    readnoiseprihdr['SUBSTRT2'] =  (                   1 , 'Starting pixel in axis 2 direction')
    readnoiseprihdr['SUBSIZE1'] =  (                2048 , 'Number of pixels in axis 1 direction')
    readnoiseprihdr['SUBSIZE2'] =  (                2048 , 'Number of pixels in axis 2 direction')
    readnoiseprihdr['FASTAXIS'] =  (                  -2 , 'Fast readout axis direction')
    readnoiseprihdr['SLOWAXIS'] =  (                  -1 , 'Slow readout axis direction')
    readnoiseprihdr['COL_STRT'] =  (                   1 , 'X-position of corner (Column)')
    readnoiseprihdr['ROW_STRT'] =  (                   1 , 'Y-position of corher (Row)')
    readnoiseprihdr['REFTYPE']  =  ('READNOISE    '      , 'Type of reference file.')
    readnoiseprihdr['AUTHOR']   =  (author)
    readnoiseprihdr['PEDIGREE'] =  (pedigree)
    readnoiseprihdr['USEAFTER'] =  (useafter)
    readnoiseprihdr['TEMP']     =  (fpa_temp             , ' FPA Temperature (K)')
    readnoiseprihdr['DESCRIP']  =  (description_readnoise)
    readnoiseprihdr.insert('TELESCOP',(''   , '' ),after=True)
    readnoiseprihdr.insert('TELESCOP',(''   , 'Instrument configuration information' ),after=True)
    readnoiseprihdr.insert('TELESCOP',(''   , '' ),after=True)
    readnoiseprihdr['HISTORY']  =  (history_readnoise)

    return readnoisehduprimary

#Function to set up primary header and DQ_DEF tables of output mask reference files
def setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise_here,thresh_cdsnoise_here,author,pedigree,useafter,fpa_temp,description_mask,history_mask):
    #Define primary HDU with standard header
    maskhduprimary = fits.PrimaryHDU()
    maskprihdr = maskhduprimary.header
    maskprihdr['DATE']    = (datetime.datetime.utcnow().isoformat() , 'Date this file was created (UTC)')
    maskprihdr['FILENAME'] =  ('niriss_ref_bad_pixel_mask.fits' , 'Name of reference file.')
    maskprihdr['DATAMODL'] =  ('MaskModel'          , 'Type of data model')
    maskprihdr['TELESCOP'] =  ('JWST    '           , 'Telescope used to acquire the data')
    maskprihdr['INSTRUME'] =  ('NIRISS  '           , 'Instrument used to acquire the data')
    maskprihdr['DETECTOR'] =  ('NIS     '           , 'Name of detector used to acquire the data')
    maskprihdr['SUBARRAY'] =  ('GENERIC '           , 'Subarray of reference file.')
    maskprihdr['SUBSTRT1'] =  (                   1 , 'Starting pixel in axis 1 direction')
    maskprihdr['SUBSTRT2'] =  (                   1 , 'Starting pixel in axis 2 direction')
    maskprihdr['SUBSIZE1'] =  (                2048 , 'Number of pixels in axis 1 direction')
    maskprihdr['SUBSIZE2'] =  (                2048 , 'Number of pixels in axis 2 direction')
    maskprihdr['FASTAXIS'] =  (                  -2 , 'Fast readout axis direction')
    maskprihdr['SLOWAXIS'] =  (                  -1 , 'Slow readout axis direction')
    maskprihdr['COL_STRT'] =  (                   1 , 'X-position of corner (Column)')
    maskprihdr['ROW_STRT'] =  (                   1 , 'Y-position of corher (Row)')
    maskprihdr['REFTYPE']  =  ('MASK    '           , 'Type of reference file.')
    maskprihdr['AUTHOR']   =  (author)
    maskprihdr['PEDIGREE'] =  (pedigree)
    maskprihdr['USEAFTER'] =  (useafter)
    maskprihdr['TEMP']     =  (              fpa_temp , ' FPA Temperature (K)')
    maskprihdr['DESCRIP']  =  (description_mask)
    maskprihdr.insert('TELESCOP',(''   , '' ),after=True)
    maskprihdr.insert('TELESCOP',(''   , 'Instrument configuration information' ),after=True)
    maskprihdr.insert('TELESCOP',(''   , '' ),after=True)
    maskprihdr['HISTORY']  =  (history_mask)

    #Define DQ_DEF binary table HDU
    flagtable = np.rec.array([
               ( 0,        1, 'DO_NOT_USE',      'Bad pixel not to be used for science or calibration'   ),
               ( 1,        2, 'DEAD',            'QE < {:6.3f} and QE of the closest 4 pixels < {:6.3f}. Not RC'.format(thresh_dead,thresh_adjopen) ),
               ( 2,        4, 'HOT',             'DC > {:6.3f} e-/s'.format(thresh_hot)   ),
               ( 3,        8, 'WARM',            'DC {:6.3f} to {:6.3f} e-/s; not marked as DO_NOT_USE because warm'.format(thresh_warm,thresh_hot)   ),
               ( 4,       16, 'LOW_QE',          'QE {:6.3f} to {:6.3f} and QE of the closest 4 pixels < {:6.3f}. Not RC'.format(thresh_dead,thresh_lowqe,thresh_adjopen)  ),
               ( 5,       32, 'RC',              'Shows RC behaviour in darks'   ),
               ( 6,       64, 'TELEGRAPH',       'Shows RTS behaviour in darks; not marked as DO_NOT_USE because RTS'   ),
               ( 7,      128, 'NON_LINEAR',      'Shows non-linearity up the ramp'    ),  
               ( 8,      256, 'BAD_REF_PIXEL',   'All bad reference pixels'   ),
               ( 9,      512, 'UNRELIABLE_BIAS', 'Bias level > {:8.1f} and/or bias noise > {:6.3f} sigma'.format(thresh_bias,thresh_bias_sigma)   ),		
               ( 10,    1024, 'UNRELIABLE_DARK', 'CDS noise > {:6.3f} times median in long darks'.format(thresh_cdsnoise_here)   ),			
               ( 11,    2048, 'UNRELIABLE_SLOPE','Slope noise > {:6.3f} times median in long darks'.format(thresh_slopenoise_here)   ),
               ( 12,    4096, 'UNRELIABLE_FLAT', 'High noise in illuminated flat-field data'   ), 
               ( 13,    8192, 'OPEN',            'QE < {:6.3f} and QE of the closest 4 pixels > {:6.3f}. Not RC'.format(thresh_lowqe,thresh_adjopen)  ),	
               ( 14,   16384, 'ADJ_OPEN',        'One of 8 pixels near open/RC with closest 4 pixels > {:6.3f} or one of 4 pixels adjacent to open/RC with closest 4 pixels {:6.3f} to {:6.3f}'.format(thresh_nearopen,thresh_adjopen,thresh_nearopen)   ), 
               ( 15,   32768, 'OTHER_BAD_PIXEL', 'Other bad pixel type'   ),    
               ( 16,   65536, 'REFERENCE_PIXEL', 'All reference pixels'   )],
               formats = 'int32,int32,a40,a140',
               names = 'Bit,Value,Name,Description')

    maskhdudqdef = fits.BinTableHDU(flagtable,name='DQ_DEF  ',ver=1)
    return maskhduprimary,maskhdudqdef,flagtable

#===============================================================
#####Main Program####

#Read in yaml config file parameters
parser = argparse.ArgumentParser()
parser.add_argument("configfile", help="Location of config file with input parameters")
args = parser.parse_args()
configfile = args.configfile
params = yaml.full_load(open(configfile, 'r'))
author = params['author']
pedigree = params['pedigree']
useafter = str(params['useafter'])
fpa_temp = params['fpa_temp']
description_readnoise = params['description_readnoise']
history_readnoise = params['history_readnoise']
description_mask = params['description_mask']
history_mask = params['history_mask']
rawdir = params['rawdir']
outdir = params['outdir']
rootreadnoise = params['rootreadnoise']
rootbpm = params['rootbpm']
refinddarkbadpixels = params['refinddarkbadpixels']
includelinearity = params['includelinearity']
reffile_superbias = params['reffile_superbias']
reffile_dark = params['reffile_dark']
reffile_flat = params['reffile_flat']
reffile_linearity = params['reffile_linearity']
gain = params['gain']
thresh_lowqe = params['thresh_lowqe']
thresh_dead = params['thresh_dead']
thresh_adjopen = params['thresh_adjopen']
thresh_nearopen = params['thresh_nearopen']
thresh_hot = params['thresh_hot']
thresh_warm = params['thresh_warm']
thresh_cold = params['thresh_cold']
thresh_bias = params['thresh_bias']
thresh_bias_sigma = params['thresh_bias_sigma']
thresh_slopenoise = params['thresh_slopenoise']
thresh_cdsnoise = params['thresh_cdsnoise']
thresh_slopenoise_longonly = params['thresh_slopenoise_longonly']
thresh_cdsnoise_shortonly = params['thresh_cdsnoise_shortonly']
thresh_slopenoise_refpix = params['thresh_slopenoise_refpix']
thresh_cdsnoise_refpix = params['thresh_cdsnoise_refpix']
reffile_bpm = params['reffile_bpm']
reffile_saturation = params['reffile_saturation']
reffile_gain = params['reffile_gain']
fraction_maxcr = params['fraction_maxcr']
sigiters_temp = params['sigiters_temp']
sigiters_final = params['sigiters_final']
runrtscheck = params['runrtscheck']
outputseparatefiles = params['outputseparatefiles']

configparamstring = '. Parameters are includelinearity = {}, reffile_superbias = {}, reffile_dark = {}, reffile_flat = {}, reffile_linearity = {}, gain = {}, thresh_lowqe = {}, thresh_dead = {}, thresh_adjopen = {}, thresh_nearopen = {}, thresh_hot = {}, thresh_warm = {}, thresh_cold = {}, thresh_bias = {}, thresh_bias_sigma = {}, thresh_slopenoise = {}, thresh_cdsnoise = {}, thresh_slopenoise_longonly= {}, thresh_cdsnoise_shortonly = {}, reffile_bpm = {}, reffile_saturation = {}, reffile_gain = {}, fraction_maxcr = {}, sigiters_temp = {}, sigiters_final = {}'.format(includelinearity, os.path.basename(reffile_superbias), os.path.basename(reffile_dark), os.path.basename(reffile_flat), os.path.basename(reffile_linearity), gain, thresh_lowqe, thresh_dead, thresh_adjopen, thresh_nearopen, thresh_hot, thresh_warm, thresh_cold, thresh_bias, thresh_bias_sigma, thresh_slopenoise, thresh_cdsnoise, thresh_slopenoise_longonly, thresh_cdsnoise_shortonly, os.path.basename(reffile_bpm) , os.path.basename(reffile_saturation), os.path.basename(reffile_gain), fraction_maxcr, sigiters_temp, sigiters_final)

history_readnoise = history_readnoise + configparamstring
history_mask = history_mask + configparamstring


#Make main output directory
if not os.path.exists(outdir):
    os.makedirs(outdir)    

#Set up logging    
logfile = datetime.datetime.now().strftime('makebpmreadnoise_%Y%m%d_%H%M%S.log')
logdirfile = os.path.join(outdir,logfile)
print ('Log is {}'.format(logdirfile))
logging.basicConfig(filename=logdirfile, filemode='w', level=logging.INFO, force=True)
logging.info('Running makebpmreadnoise.py with config: {}'.format(configfile))

#Set location of savezfile that saves some intermediate arrays for darks
savezfile = os.path.join(outdir,'saveddarkarrays.npz')

#Define reference file locations and output file locations and names
readnoisefile = os.path.join(outdir,(rootreadnoise+'.fits'))
rootbpmfile = os.path.join(outdir,rootbpm)
longbpmfile = rootbpmfile + '_long.fits'
shortbpmfile = rootbpmfile + '_short.fits'
minimalbpmfile = rootbpmfile + '_minimal.fits'
typebpmdir = os.path.join(outdir,'badpixeltypes')
typebpmfile = os.path.join(typebpmdir,(rootbpm+'_type'))

#Get number of raw dark files
rawfiles = glob.glob(os.path.join(rawdir,'*uncal.fits'))
numraw = len(rawfiles)
logging.info('Found {} raw dark files in {}'.format(numraw,rawdir))
print ('Found {} raw dark files in {}'.format(numraw,rawdir))
listrawfiles = '. Input raw dark files: '
for file in rawfiles:
    if file == rawfiles[0]:
        listrawfiles = '. Input raw dark files: {}'.format(os.path.basename(file))
    else:
        listrawfiles = listrawfiles +',{}'.format(os.path.basename(file))
listrawfiles = listrawfiles + '.'        
print (listrawfiles)

history_readnoise = history_readnoise + listrawfiles
history_mask = history_mask + listrawfiles

#Expect processed directories to have names as produced by runDetector1_niriss_fullramedarks.py
datadir = os.path.dirname(rawdir)
slopedir = os.path.join(datadir,'rate')
cubedir = os.path.join(datadir,'jump')

#Make output directories if they do not exist
outcubedir = os.path.join(datadir,'cubefixcr')
noisedir = os.path.join(datadir,'noise')
tempnoisedir = os.path.join(datadir,'tempnoise')
if not os.path.exists(outcubedir):
    os.makedirs(outcubedir)
if not os.path.exists(noisedir):
    os.makedirs(noisedir)
if not os.path.exists(tempnoisedir):
    os.makedirs(tempnoisedir)

#=======================================================================
#Create empty arrays for all masks - some of these separate for short, long and minimal bpms
arrshape = (2048,2048)
bpm = np.zeros(arrshape, dtype='int32')
donotuseshort = np.zeros(arrshape, dtype = 'uint8')
donotuselong = np.zeros(arrshape, dtype='uint8')
donotuseminimal = np.zeros(arrshape, dtype='uint8')
dead = np.zeros(arrshape, dtype='uint8')
hot = np.zeros(arrshape, dtype='uint8')
warm = np.zeros(arrshape, dtype='uint8')
lowqe = np.zeros(arrshape, dtype='uint8')
rc = np.zeros(arrshape, dtype='uint8')
telegraph = np.zeros(arrshape, dtype='uint8')
#nonlinear = np.zeros(arrshape, dtype='uint8')
#nosatcheck = np.zeros(arrshape, dtype='uint8')
badrefpixelshort = np.zeros(arrshape, dtype='uint8')
badrefpixellong = np.zeros(arrshape, dtype='uint8')
badrefpixelminimal = np.zeros(arrshape, dtype='uint8')
unrelbias = np.zeros(arrshape, dtype='uint8')
unreldarkshort = np.zeros(arrshape, dtype='uint8')
unreldarklong = np.zeros(arrshape, dtype='uint8')
unreldarkminimal = np.zeros(arrshape, dtype='uint8')
unrelslopeshort = np.zeros(arrshape, dtype='uint8')
unrelslopelong = np.zeros(arrshape, dtype='uint8')
unrelslopeminimal = np.zeros(arrshape, dtype='uint8')
unrelflat = np.zeros(arrshape, dtype='uint8')
openpixel = np.zeros(arrshape, dtype='uint8')
adjopen = np.zeros(arrshape, dtype='uint8')
otherbadshort = np.zeros(arrshape, dtype='uint8')
otherbadlong = np.zeros(arrshape, dtype='uint8')
otherbadminimal = np.zeros(arrshape, dtype='uint8')
refpixel = np.ones(arrshape, dtype='uint8')
noisyindarks = np.zeros(arrshape, dtype='uint8')

#Identify only reference pixels with value 1
refpixel[4:2044,4:2044] = 0

#numimages = 60 #Trim to 60 files cos looks to be too large an array for crhits with all 65.

#=======================================================================
#Do this block if have not already run through the darks to identify all bad pixels.
#Will be skipped if refinddarkbadpixels = False.
if refinddarkbadpixels==True:

    #Make required intermediate reference files and process data, if necessary
    #Check for superbias reference file and make it if it doesn't exist
    if not os.path.exists(reffile_superbias):
        logging.info('Superbias reference file {} does not exist. Making it with makesuperbias_jwst_reffiles.py'.format(reffile_superbias))
        run_superbias(rawfiles,reffile_superbias,author,pedigree,useafter)
    else:
        logging.info('Found superbias reference file to use: {}'.format(reffile_superbias))

    #Check to see what processing of darks exists and run through pipeline if needed
    rootdir = os.path.dirname(rawdir)
    dirlist=natsort.natsorted(os.listdir(rawdir))
    dirlist[:] = (value for value in dirlist if value.endswith('_uncal.fits'))
    numimages = len(dirlist)
    print (numimages)
    #numimages=1
    #loop over exposures
    dopipe=True
    if dopipe==True:
        #for j in range(numimages):
        for j in range(60,numimages):
            rawdirfile = os.path.join(rawdir,dirlist[j])
            print (rawdirfile,rootdir,reffile_bpm,reffile_saturation,reffile_superbias,reffile_linearity,reffile_gain)
            result = det1_pipeline_onefile(rawdirfile,rootdir,reffile_bpm,reffile_saturation,reffile_superbias,reffile_linearity,reffile_gain)
    
    #old code that ran all raw images through level 1 pipeline 
    #if ((os.path.exists(slopedir))&(os.path.exists(cubedir))):
    #    slopefiles = glob.glob(os.path.join(slopedir,'*rate.fits'))
    #    cubefiles = glob.glob(os.path.join(cubedir,'*.fits'))
    #    numslope = len(slopefiles)
    #    numcube = len(cubefiles)
    #    if ((numslope != numraw)|(numcube != numraw)):
    #        logging.info('Level 1 processed cube and slope data does not exist. Making it with runDetector1_niriss_fullframedark.py')
    #        result = det1_pipeline(rawdir,reffile_bpm,reffile_saturation,reffile_superbias,reffile_linearity,reffile_gain)
    #else:
    #    logging.info('Level 1 processed cube and slope data does not exist. Making it with runDetector1_niriss_fullframedark.py')
    #    result = det1_pipeline(rawdir,reffile_bpm,reffile_saturation,reffile_superbias,reffile_linearity,reffile_gain)

    #Check for dark current reference file and make it if it doesn't exist
    cubefiles = glob.glob(os.path.join(cubedir,'*.fits'))
    if not os.path.exists(reffile_dark):
        logging.info('Dark current reference file {} does not exist. Making it with jwst_reffiles.dark_current'.format(reffile_dark))
        dark_reffile.Dark(file_list=cubefiles,output_file=reffile_dark, author=author, pedigree=pedigree, use_after=useafter, descrip='This is a dark current reference file', history='Created with jwst_reffiles', sigma_threshold=3)
    else:
        logging.info('Found dark current reference file to use: {}'.format(reffile_dark))

    #Dark noise files have to made twice, first using original data, then later with cubes corrected for cosmic rays
    #Check to see if initial temporary (before fully corrected for cosmic rays) dark noise files exist
    #If not then make them
    noiserootout = 'temp'
    tempslopefileroot = noiserootout + 'sigmadarkzero.fits'
    tempslopenoisefile = os.path.join(tempnoisedir,tempslopefileroot)
    tempcdsnoisefile = tempslopenoisefile.replace('sigmadarkzero.fits','mediancds1fstd.fits')
    tempslopenoisefilelist = glob.glob(os.path.join(tempnoisedir,'*sigmadarkzero.fits'))
    tempcdsnoisefilelist = glob.glob(os.path.join(tempnoisedir,'*mediancds1fstd.fits'))

    if ((len(tempslopenoisefilelist)==1) and (len(tempcdsnoisefilelist)==1)):
        print ('Found existing initial slope and CDS noise files:')
        print ('Slope noise: {}'.format(tempslopenoisefilelist[0]))
        print ('CDS noise: {}'.format(tempcdsnoisefilelist[0]))
        logging.info('Found existing initial slope and CDS noise files:') 
        logging.info('Slope noise: {}'.format(tempslopenoisefilelist[0]))
        logging.info('CDS noise: {}'.format(tempcdsnoisefilelist[0]))
    else:    
        makenoisescmd = './makedarknoisefilesgdq.py {} {} {} {} {} --sigiters {} --logstats'.format(cubedir,slopedir,tempnoisedir,noiserootout,fraction_maxcr,sigiters_temp)
        print ('Making initial temporary slope and CDS noise files with command:')
        print ('{}'.format(makenoisescmd))
        logging.info('Making initial temporary slope and CDS noise files with command:')
        logging.info('{}'.format(makenoisescmd))
        os.system(makenoisescmd)

    #======================================================
    #Read in initial noise files and make some statistics

    #Read in initial temporary noise files
    hdulist = fits.open(tempslopenoisefile)
    header = hdulist[0].header
    slopenoise = hdulist[0].data*gain
    hdulist = fits.open(tempcdsnoisefile)
    cdsnoise = hdulist[0].data*gain

    #Get statistics of initial temporary noise separately for reference and active pixels
    a = np.where(refpixel==0)
    slopenoiseactive = slopenoise[a]
    cdsnoiseactive = cdsnoise[a]
    r = np.where(refpixel==1)
    slopenoiserefpix = slopenoise[r]
    cdsnoiserefpix = cdsnoise[r]

    logging.info('')
    logging.info('Stats of slope and CDS noise files')

    imstatslopenoiseactive = imagestats.ImageStats(slopenoiseactive,fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('Slope noise (non-reference pixels):  Mean: {}, Median: {}, Std dev: {}'.format(imstatslopenoiseactive.mean,imstatslopenoiseactive.median,imstatslopenoiseactive.stddev))
    imstatcdsnoiseactive = imagestats.ImageStats(cdsnoiseactive,fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
    logging.info('CDS noise (non-reference pixels):  Mean: {}, Median: {}, Std dev: {}'.format(imstatcdsnoiseactive.mean,imstatcdsnoiseactive.median,imstatcdsnoiseactive.stddev))
    imstatslopenoiserefpix = imagestats.ImageStats(slopenoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
    imstatcdsnoiserefpix = imagestats.ImageStats(cdsnoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
    logging.info('')

    #=====================================================================================
    #Start identifying and classifying some bad pixels from darks and look for cosmic rays
    print ('Start identifying and classifying some bad pixels from darks and look for cosmic rays')

    #Set thresholds used for finding RC and inverse RC pixels
    #Set excess in first 3 CDS points in the ramp at 7 times median
    first3excesslimit = 7.0*imstatcdsnoiseactive.median 
    logging.info('first3excesslimit for finding RC and inverse RC pixels: {}'.format(first3excesslimit))

    #Identify pixels noisy in darks that require checking and typing
    w = np.where(((cdsnoise>=(1.5*imstatcdsnoiseactive.median))&(refpixel==0))|((cdsnoise>=(1.5*imstatcdsnoiserefpix.median))&(refpixel==1))|((slopenoise>=(1.5*imstatslopenoiseactive.median))&(refpixel==0))|((slopenoise>=(1.5*imstatslopenoiserefpix.median))&(refpixel==1)))
    noisyindarks[w] = 1

    #Use CDS files to get time ordered sequences of pixels
    cubedirlist = natsort.natsorted(os.listdir(cubedir))
    cubedirlist[:] = (value for value in cubedirlist if value.endswith('.fits'))
    #numimages = len(cubedirlist)
    numimages = 60 #Trim to 60 files cos looks to be too large an array for crhits with all 65.
    logging.info('Using {} CDS images to build pixel time series'.format(numimages))

    #Will read in strips of data to avoid having all images in memory at once
    dxstrip = 100
    numsec = int(2048/dxstrip+1)
    numpixsec = 2048*numsec
    maxx = int(np.min([(numsec*dxstrip,2048)]))

    for i in range(numsec):
        xmin = i*dxstrip
        xmax = min([2048,xmin+dxstrip])
        print ('loading cube section {} to {} at time {}'.format(xmin,xmax,datetime.datetime.now()))
        logging.info('loading cube section {} to {}'.format(xmin,xmax))
        for j in range(numimages):
            cubefile = os.path.join(cubedir,cubedirlist[j])
            with fits.open(cubefile,memmap=True) as hdul:
                header = hdul[0].header
                ngroup = header['NGROUPS']
                cds = np.diff(np.squeeze(hdul['SCI'].data[:,:,:,xmin:xmax]),axis=0)*gain
            #Change to type integer to save memory    
            cdsinteger = np.rint(cds)
            cds = cdsinteger.astype(int)
            if j==0:
                cdstime = cds
            else:
                cdstime = np.vstack((cdstime,cds))
            del cdsinteger,cds
        print ('loaded cube section {} to {} at time {}'.format(xmin,xmax,datetime.datetime.now()))

        #Get same section of noise files, bpm and make a mask to contain cr hits
        noisyindarkssection = noisyindarks[:,xmin:xmax]
        cdsnoisesection = cdsnoise[:,xmin:xmax]
        slopenoisesection = slopenoise[:,xmin:xmax]    
        numsigcds = cdsnoisesection/imstatcdsnoiseactive.median
        crhitssection = np.zeros(cdstime.shape, dtype='uint8')
        noncrhitssection = np.zeros(cdstime.shape, dtype='uint8')

        #Determine first 3 CDS excess statistic using median per CDS for all exposures
        xesgroup1list2d = [(np.array([0])+p*(ngroup-1)) for p in range(numimages)]
        xesgroup2list2d = [(np.array([1])+p*(ngroup-1)) for p in range(numimages)]
        xesgroup3list2d = [(np.array([2])+p*(ngroup-1)) for p in range(numimages)]
        #Use comparison groups at end of exposure 
        xeslaterlist2d = [(np.arange((ngroup-30),(ngroup-3))+p*(ngroup-1)) for p in range(numimages)]
        xesgroup1list = np.ravel(xesgroup1list2d)
        xesgroup2list = np.ravel(xesgroup2list2d)
        xesgroup3list = np.ravel(xesgroup3list2d)
        numxesgroup1list = xesgroup1list.size
        xeslaterlist = np.ravel(xeslaterlist2d)
        xesgroup1median = (np.median(cdstime[xesgroup1list,:,:],axis=0))
        xesgroup2median = (np.median(cdstime[xesgroup2list,:,:],axis=0))
        xesgroup3median = (np.median(cdstime[xesgroup3list,:,:],axis=0))
        #Get excess compared to the typical CDS value at the end of the exposure
        first3excess = (xesgroup1median+xesgroup2median+xesgroup3median)-3.0*(np.median(cdstime[xeslaterlist,:,:],axis=0))
        first3excess1d = np.ravel(first3excess) 

        #Select RC pixels as first 3 CDS excess greater than the threshold
        w = np.where(first3excess > first3excesslimit)
        xfull = xmin+w[1]
        rc[w[0],xfull] = 1
        numrc = first3excess[w].size

        #Select 'Inverse RC' pixels as first 3 CDS excess lower than the threshold and flag as "OTHER_BAD" because there is no category for Inverse RC
        w = np.where(first3excess < (-1.0*first3excesslimit))
        xfull = xmin+w[1]
        otherbadshort[w[0],xfull] = 1
        otherbadlong[w[0],xfull] = 1
        otherbadminimal[w[0],xfull] = 1
        numinvrc = first3excess[w].size

        #Select possible Random Telegraph Signal (RTS) pixels to be checked for cosmic rays and RTS behaviour
        w = np.where((noisyindarkssection==1)&(first3excess <= first3excesslimit)&(first3excess >= (-1.0*first3excesslimit)))
        w0 = w[0]
        w1 = w[1]
        numpossiblerts = w0.size
        print ('numrc={}, numinvrc={},numpossiblerts={},numnoisy={}'.format(numrc,numinvrc,numpossiblerts,noisyindarkssection[(np.where(noisyindarkssection==1))].size))
        logging.info('numrc={}, numinvrc={},numpossiblerts={},numnoisy={}'.format(numrc,numinvrc,numpossiblerts,noisyindarkssection[(np.where(noisyindarkssection==1))].size))

        #Iterate over each pixel
        for x in range(numpossiblerts):
            xfull = xmin+w1[x]
            yfull = w0[x]

            #Remove most cosmic rays by iteratively clipping positive CDS values when no negative values within 80% (RTS pixels typically have symmetric -ve and +ve jumps). 
            #Subtract median to avoid finding lots of 'cosmic rays' in hot pixels
            cdstimesortindex = np.argsort(cdstime[:,w0[x],w1[x]])
            cdstimesorted = cdstime[cdstimesortindex,w0[x],w1[x]]-np.median(cdstime[:,w0[x],w1[x]])
            while (cdstimesorted[0]> (-0.8*cdstimesorted[-1])):
                crhitssection[cdstimesortindex[-1],w0[x],w1[x]] = 1
                cdstimesorted = cdstimesorted[:-1]
                cdstimesortindex = cdstimesortindex[:-1]

            #RTS flagging is only informational and does not change the number of pixels flagged as DO_NOT_USE.
            #It is time-consuming and therefore optional.
            if runrtscheck:

                #Set up CDS bins, including +/-20 beyond last point.
                binmax = 2*round((np.amax(cdstimesorted)+1)/2)+21.5
                binmin = 2*round((np.amin(cdstimesorted)-1)/2)-20.5
                numbins = int((binmax-binmin+1)/2.0)
                binedges = (np.arange(numbins+1)*(binmax-binmin)/numbins)+binmin
                bincen = binedges[:-1]+1.0
                histcds,edges = np.histogram(cdstimesorted,binedges)
                histcdserr = histcds**0.5
                histcdserr[np.where(histcds==0)] = 1.0

                #Fit three constrained Gaussians to the CDS histogram
                #a1 - peak of central Gaussian, should be close to zero for darks
                a1init = 0.0
                #a2 - offset from a1 of other two peaks, initially set to 35% of max-min value
                a2init = 0.35*(binmax-binmin)
                #a3 - height of central Gaussian, initially set to max of histogram
                a3init = np.amax(histcds)
                #a4 - height of other 2 Gaussians, variable depending upon RTS frequency
                a4init = 0.3*a3init
                #a5 - dispersion of all 3 Gaussians - initially set to median CDS noise
                a5init = imstatcdsnoiseactive.median
                lnfinit = -3.0
                paramsinit = [a1init,a2init,a3init,a4init,a5init,lnfinit]

                #Find the maximum likelihood value. 
                #Set bounds on a4init so secondary peak at least 2% of main peak.
                #Set bounds on a5init so sigma of gaussians similar to CDS noise
                chi2 = lambda *args: -2 * lnlike(*args)
                result = op.minimize(chi2, paramsinit, args=(bincen, histcds, histcdserr), method='SLSQP', bounds=((None, None), (15.0, None), (0.0, None), (0.02*a3init, None), (0.7*imstatcdsnoiseactive.median, 1.8*imstatcdsnoiseactive.median), (None, None)))
                a1_ml, a2_ml, a3_ml, a4_ml, a5_ml, lnf_ml = result["x"]

                maxlichisq = result["fun"]
                maxliresult = result["x"]
                maxliresult[4] = np.absolute(maxliresult[4])

                bestmodel = a3_ml*np.exp((-1.0*(bincen-a1_ml)**2.0)/(2.0*a5_ml**2.0))+a4_ml*np.exp((-1.0*(bincen-(a1_ml-a2_ml))**2.0)/(2.0*a5_ml**2.0))+a4_ml*np.exp((-1.0*(bincen-(a1_ml+a2_ml))**2.0)/(2.0*a5_ml**2.0))

                #Fit single Gaussian for comparison using measured CDS noise for this pixel
                a1singleinit = 0.0
                a3singleinit = np.amax(histcds)
                a5singleinit = cdsnoisesection[w0[x],w1[x]]
                lnfsingleinit = -3.0
                singleparamsinit = [a1singleinit,a3singleinit,a5singleinit,lnfsingleinit]

                #Find the maximum likelihood value.
                chi2single = lambda *args: -2 * lnlikesingle(*args)
                resultsingle  = op.minimize(chi2single, singleparamsinit, args=(bincen, histcds, histcdserr), method='SLSQP', bounds=((None, None), (0.0, None), (0.7*imstatcdsnoiseactive.median, None), (None, None)))
                a1single_ml, a3single_ml, a5single_ml, lnf_ml = resultsingle["x"]
                maxlichisqsingle = resultsingle["fun"]
                maxliresultsingle = resultsingle["x"]
                bestmodelsingle = a3single_ml*np.exp((-1.0*(bincen-a1single_ml)**2.0)/(2.0*a5single_ml**2.0))

                #Check if BIC shows extra parameters are justified
                #BIC = -2 ln L + k*(ln n - ln 2pi) where L lis likeli, k is num param, n is numbins
                bicthreehist = maxlichisq+5.0*(math.log(numbins)-math.log(6.28))
                bicsinglehist = maxlichisqsingle+3.0*(math.log(numbins)-math.log(6.28))
                bicdiff = (bicsinglehist-bicthreehist)

                a3mlfrac = a3_ml/(a3_ml+2.0*a4_ml)
                a4mlfrac = 1.0-a3mlfrac 

                #If BIC difference > 20 and fit is well-constrained, flag as telegraph
                if ((a4mlfrac>0.005)&(a4mlfrac<0.99)&(bicdiff>20)):
                    telegraph[yfull,xfull] = 1

        del cdstime

        #Set up arrays for locations of crhits and non-crhits in noisy pixels 
        if 'crhits' not in locals():
            crhits = np.zeros((((ngroup-1)*numimages),2048,2048), dtype='bool')
            noncrhits = np.zeros((((ngroup-1)*numimages),2048,2048), dtype='bool')
        crhits[:,:,xmin:xmax] = crhitssection
        logging.info('')
        logging.info('Removed {} cosmic ray hits from section'.format( crhitssection[np.where(crhitssection==1)].size))
        logging.info('')
        #First set all noisy pixels in noncrhits to true 
        noncrhitssection[:,w0,w1] = True

        noncrhits[:,:,xmin:xmax] = noncrhitssection

        del noisyindarkssection,cdsnoisesection,slopenoisesection,numsigcds,crhitssection,noncrhitssection

    #Unset all pixels containing cosmic rays from noncrhits
    noncrhits[np.where(crhits==True)] = False

    #=====================================================================================
    #Open dark cubes and for pixels that were checked because possible RTS noise edit their GROUPDQ to:
    #1. remove jump flags that are probably not cosmic rays
    #2. add jump flags when detected by this routine in case they were not found by jumpstep
    for j in range(numimages):

        cubefile = os.path.join(cubedir,cubedirlist[j])
        outcubefile = os.path.join(outcubedir,cubedirlist[j])
        outcubefile = outcubefile.replace('.fits','_fixcr.fits')

        #Only make new cube file if outcubefile does not already exist
        if not os.path.exists(outcubefile):
            print ('Updating GROUPDQ for cosmic rays in dark cube file {}'.format(outcubefile))
            logging.info('Updating GROUPDQ for cosmic rays in dark cube file {}'.format(outcubefile))
            crhitsthisimage = crhits[j*(ngroup-1):(j+1)*(ngroup-1),:,:]
            crhitsthisimage = np.pad(crhitsthisimage,((1,0),(0,0),(0,0)), 'minimum')
            noncrhitsthisimage = noncrhits[j*(ngroup-1):(j+1)*(ngroup-1),:,:]
            noncrhitsthisimage = np.pad(noncrhitsthisimage,((1,0),(0,0),(0,0)), 'maximum')

            with fits.open(cubefile) as hdul:
                gdq = hdul['GROUPDQ'].data

                w = np.where(gdq[0,:,:,:]==dqflags.group['JUMP_DET'])
                logging.info('total of %s jump flags before changes' %  (gdq[0,w[0],w[1],w[2]].size))

                #Remove jump flags that are probably not cosmic rays
                w = np.where((noncrhitsthisimage==True)&(gdq[0,:,:,:]==dqflags.group['JUMP_DET']))
                gdq[0,w[0],w[1],w[2]]  = gdq[0,w[0],w[1],w[2]] -  dqflags.group['JUMP_DET']
                logging.info('removed %s jump flags' %  (gdq[0,w[0],w[1],w[2]].size))

                #Add jump flags when detected by this routine in case they were not found by jumpstep
                w = np.where(crhitsthisimage==True)
                gdq[0,w[0],w[1],w[2]]  = np.bitwise_or(gdq[0,w[0],w[1],w[2]], dqflags.group['JUMP_DET'])
                logging.info('added %s jump flags' %  (gdq[0,w[0],w[1],w[2]].size))

                #Add in neighbours of all large jumps (>3000) in case they were removed
                cds = np.diff(np.squeeze(hdul['SCI'].data),axis=0)
                w = np.where((cds>3000)&(gdq[0,1:,:,:]==dqflags.group['JUMP_DET']))
                gdq[0,w[0]+1,w[1]+1,w[2]]  = np.bitwise_or(gdq[0,w[0]+1,w[1]+1,w[2]], dqflags.group['JUMP_DET'])
                gdq[0,w[0]+1,w[1]-1,w[2]]  = np.bitwise_or(gdq[0,w[0]+1,w[1]-1,w[2]], dqflags.group['JUMP_DET'])
                gdq[0,w[0]+1,w[1],w[2]+1]  = np.bitwise_or(gdq[0,w[0]+1,w[1],w[2]+1], dqflags.group['JUMP_DET'])
                gdq[0,w[0]+1,w[1],w[2]-1]  = np.bitwise_or(gdq[0,w[0]+1,w[1],w[2]-1], dqflags.group['JUMP_DET'])

                hdul['GROUPDQ'].data = gdq
                hdul.writeto(outcubefile,overwrite=True)
        else:
            print ('Skipping making edited dark cube file {} because already exists'.format(outcubefile))
            logging.info('Skipping making edited dark cube file {} because already exists'.format(outcubefile))

    #save numpy arrays here in case need them later running with refinddarkbadpixels = False
    np.savez(savezfile, rc=rc, otherbadshort=otherbadshort, otherbadlong=otherbadlong, otherbadminimal=otherbadminimal,telegraph=telegraph)
    logging.info('Saved numpy arrays in {}'.format(savezfile)) 
    logging.info('')
    
else:
    #If refinddarkbadpixels = False, load the saved arrays from the section above
    npzloaded = np.load(savezfile)
    rc              = npzloaded['rc']
    otherbadshort   = npzloaded['otherbadshort']
    otherbadlong    = npzloaded['otherbadlong']
    otherbadminimal = npzloaded['otherbadminimal']
    telegraph       = npzloaded['telegraph']
    logging.info('Loaded numpy arrays from {}'.format(savezfile)) 
    logging.info('')
    
#=====================================================================================
#Run makedarknoisefilesgdq.py again now that GROUPDQ have been updated, unless output files already exist

noiserootout = 'gdq'
slopefileroot = noiserootout + 'sigmadarkzero.fits'
slopenoisefile = os.path.join(noisedir,slopefileroot)
cdsnoisefile = slopenoisefile.replace('sigmadarkzero.fits','mediancdsstd.fits')
cds1fnoisefile = slopenoisefile.replace('sigmadarkzero.fits','mediancds1fstd.fits')
slopenoisefilelist = glob.glob(os.path.join(noisedir,'*sigmadarkzero.fits'))
cdsnoisefilelist = glob.glob(os.path.join(noisedir,'*mediancdsstd.fits'))
cds1fnoisefilelist = glob.glob(os.path.join(noisedir,'*mediancds1fstd.fits'))
logging.info('')

if ((len(slopenoisefilelist)==1) and (len(cdsnoisefilelist)==1)):
    print ('Found existing groupDQ-fixed slope and CDS noise files:')
    print ('Slope noise: {}'.format(slopenoisefilelist[0]))
    print ('CDS noise: {}, '.format(cdsnoisefilelist[0]))
    print ('CDS 1/f-subtracted noise: {}'.format(cds1fnoisefilelist[0]))
    logging.info('Found existing groupDQ-fixed slope and CDS noise files:')
    logging.info('Slope noise: {}'.format(slopenoisefilelist[0]))
    logging.info('CDS noise: {}'.format(cdsnoisefilelist[0]))
    logging.info('CDS 1/f-subtracted noise: {}'.format(cds1fnoisefilelist[0]))
else:    
    makenoisescmd = './makedarknoisefilesgdq.py {} {} {} {} {} --sigiters {} --reffile_dark {} --usegdq --logstats'.format(outcubedir,slopedir,noisedir,noiserootout,fraction_maxcr,sigiters_final,reffile_dark)
    print ('Making groupDQ-fixed slope and CDS noise files with command:')
    print ('{}'.format(makenoisescmd))
    logging.info('Making groupDQ-fixed slope and CDS noise files with command:')
    logging.info ('{}'.format(makenoisescmd))
    os.system(makenoisescmd)

logging.info('')
    

#=====================================================================================
#Load final run of dark noise data (after fully corrected for cosmic rays) to identify and classify more bad pixels

#Get info from a raw file header
rawhead = fits.getheader(rawfiles[0],'PRIMARY')
readpattern = rawhead['READPATT']

#Load slope noise file and multiply by gain to get in e-
hdulist = fits.open(slopenoisefile)
slopenoise = hdulist[0].data*gain

#Load dark current file
darkcurrfile = slopenoisefile.replace('sigmadarkzero.fits','mediandark.fits')
hdulist = fits.open(darkcurrfile)
darkcurr = hdulist[0].data*gain

#Load cds noise (without 1/f correction) file - this is the noise for the readnoise reference file in units of ADU 
hdulist = fits.open(cdsnoisefile)
readnoise = hdulist[0].data

#Load cds noise (with 1/f correction) file - this is the noise for the bad pixels analysis, multiply by gain for units of electrons.
hdulist = fits.open(cds1fnoisefile)
cdsnoise = hdulist[0].data*gain

#Load superbias file containing bias and its error - made with jwst_reffiles package
hdulist = fits.open(reffile_superbias)
superbias = hdulist['SCI'].data
superbiassigma = hdulist['ERR'].data

#Recalculate noises here...
#Get statistics separately for reference and active pixels
a = np.where(refpixel==0)
slopenoiseactive = slopenoise[a]
cdsnoiseactive = cdsnoise[a]
darkcurractive = darkcurr[a]
r = np.where(refpixel==1)
slopenoiserefpix = slopenoise[r]
cdsnoiserefpix = cdsnoise[r]

logging.info('')
logging.info('Stats of dark current, slope and CDS noise files')

imstatdarkcurractive = imagestats.ImageStats(darkcurractive,fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
logging.info('Dark current (non-reference pixels):  Mean: {}, Median: {}, Std dev: {}'.format(imstatdarkcurractive.mean,imstatdarkcurractive.median,imstatdarkcurractive.stddev))
imstatslopenoiseactive = imagestats.ImageStats(slopenoiseactive,fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
logging.info('Slope noise (non-reference pixels):  Mean: {}, Median: {}, Std dev: {}'.format(imstatslopenoiseactive.mean,imstatslopenoiseactive.median,imstatslopenoiseactive.stddev))
imstatcdsnoiseactive = imagestats.ImageStats(cdsnoiseactive,fields="npix,min,max,median,mean,stddev",nclip=3,lsig=3.0,usig=3.0,binwidth=0.1)
logging.info('CDS noise (non-reference pixels):  Mean: {}, Median: {}, Std dev: {}'.format(imstatcdsnoiseactive.mean,imstatcdsnoiseactive.median,imstatcdsnoiseactive.stddev))
imstatslopenoiserefpix = imagestats.ImageStats(slopenoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
imstatcdsnoiserefpix = imagestats.ImageStats(cdsnoiserefpix,fields="npix,min,max,median,mean,stddev",binwidth=0.1,nclip=3)
logging.info('')

#All high superbias pixels
logging.info('Unreliable bias if bias > {} ADU'.format(thresh_bias))
w=np.where(superbias>thresh_bias)
unrelbias[w] = 1

#All variable superbias pixels 
logging.info('Unreliable bias if bias rms > {} ADU'.format(thresh_bias_sigma*(np.median(superbiassigma))))
w = np.where(superbiassigma>thresh_bias_sigma*(np.median(superbiassigma)))
unrelbias[w] = 1
#Make sure includes full last column of reference pixels for NIRISS
unrelbias[:,2047] = 1

#Mask very high dark current (e.g. >0.5 e-/s, ~13x median). Below this level some are neighbours of hot pixels with well-behaved noise.
logging.info('Hot pixel if dark current > {} e-/s'.format(thresh_hot))
w = np.where(darkcurr>=thresh_hot)
hot[w] = 1

#Add warm pixels to mask but do not flag as DO_NOT_USE
logging.info('Warm pixel if dark current > {} and < {} e-/s'.format(thresh_warm,thresh_hot))
w = np.where((darkcurr>=thresh_warm)&(darkcurr<thresh_hot))
warm[w] = 1

#Mask very low dark current (cold, e.g. <0.005 e-/s).
#Most of these are also other noisy types, but some are adjacent to hot pixels and do not show in flats or noise
logging.info('Other bad pixel if non-reference pixel and very low dark current < {} e-/s'.format(thresh_cold))
w = np.where((darkcurr<=thresh_cold)&(refpixel==0))
otherbadshort[w] = 1
otherbadlong[w] = 1
otherbadminimal[w] = 1

#Mask very noisy in slope or cds for both bpm types 
#For reference pixels use lower threshold for all
logging.info('Unreliable slope if slope noise > {} (non-reference pixel) or {} (reference pixel) e-/s'.format((thresh_slopenoise*imstatslopenoiseactive.median),(thresh_slopenoise_refpix*imstatslopenoiserefpix.median)))
w = np.where(((slopenoise>=(thresh_slopenoise*imstatslopenoiseactive.median))&(refpixel==0))|((slopenoise>=(thresh_slopenoise_refpix*imstatslopenoiserefpix.median))&(refpixel==1)))
unrelslopelong[w] = 1
unrelslopeshort[w] = 1
unrelslopeminimal[w] = 1
logging.info('Unreliable dark if CDS noise > {} (non-reference pixel) or {} (reference pixel) e-/s'.format((thresh_cdsnoise*imstatcdsnoiseactive.median),(thresh_cdsnoise_refpix*imstatcdsnoiserefpix.median)))
w = np.where(((cdsnoise>=(thresh_cdsnoise*imstatcdsnoiseactive.median))&(refpixel==0))|((cdsnoise>=(thresh_cdsnoise_refpix*imstatcdsnoiserefpix.median))&(refpixel==1)))
unreldarkshort[w] = 1
unreldarklong[w] = 1
unreldarkminimal[w] = 1

#Mask less noisy in slope only for long ramp bpm
logging.info('For long ramp BPM only, unreliable slope if slope noise > {} (non-reference pixel) or {} (reference pixel) e-/s'.format((thresh_slopenoise_longonly*imstatslopenoiseactive.median),(thresh_slopenoise_longonly*imstatslopenoiserefpix.median)))
w = np.where(((slopenoise>=(thresh_slopenoise_longonly*imstatslopenoiseactive.median))&(refpixel==0))|((slopenoise>=(thresh_slopenoise_longonly*imstatslopenoiserefpix.median))&(refpixel==1)))
unrelslopelong[w] = 1

#Mask less noisy in cds only for short ramp bpm
logging.info('For short ramp BPM only, unreliable dark if CDS noise > {} (non-reference pixel) or {} (reference pixel) e-/s'.format((thresh_cdsnoise_shortonly*imstatcdsnoiseactive.median),(thresh_cdsnoise_shortonly*imstatcdsnoiserefpix.median)))
w = np.where(((cdsnoise>=(thresh_cdsnoise_shortonly*imstatcdsnoiseactive.median))&(refpixel==0))|((cdsnoise>=(thresh_cdsnoise_shortonly*imstatcdsnoiserefpix.median))&(refpixel==1)))
unreldarkshort[w] = 1


#=====================================================================================
#Add nonlinear pixels to BPM (not necessarily DO_NOT_USE)

#Load linearity reference file DQ array
#Deprecated
#hdulist = fits.open(reffile_linearity)
#lindq = hdulist['DQ'].data
#lindqdef=Table(hdulist['DQ_DEF'].data)
#nonlinflag=lindqdef['Value'][np.where(lindqdef['Name']=='NONLINEAR')].data[0]
#nonlinear[np.where((np.bitwise_and(lindq, nonlinflag) == 1))] = 1
#logging.info('nonlinear set from linearity reference file')

#=====================================================================================
#Use illuminated flat field images to identify and classify more bad pixels

#Load grism flat field file
hdulist = fits.open(reffile_flat)
flat1 = hdulist['SCI'].data

#Deprecated - now in flat field ref files 
#UNRELIABLE_FLAT for all pixels at active pixel border because 10% higher in flats
#Sections [4:5,4:2044], [2043:2044,4:2044], [5:2043,4:5], [5:2043,2043:2044]
#logging.info('Unreliable flat set for all border pixels')
#unrelflat[4:5,4:2044] = 1
#unrelflat[2043:2044,4:2044] = 1
#unrelflat[5:2043,4:5] = 1
#unrelflat[5:2043,2043:2044] = 1

#Make version of flat without high or low pixels to use in determining background for bad neighbours
#Make unity version of flat to use as area because area function does not account for out of boundary
flat1unity = np.ones(flat1.shape)
flat1clipped = deepcopy(flat1)
v = np.where((flat1<thresh_lowqe)|(flat1>1.2))
flat1clipped[v] = 1.0

#Check to see if an open pixel with bad neighbours or low QE or dead
#Include all RC pixels in this check because they can also give flux to neighbours
#One of 8 pixels adjacent to open pixel with closest 4 pixels > 1.1 or 4 pixels adjacent to open pixel with closest 4 pixels in range 1.05 to 1.1
#Normalize neighbours by local QE

w = np.where((flat1<=thresh_lowqe)|(rc==1))
numlo = w[0].size
ally = w[0]
allx = w[1]

for k in range(numlo):
    y = ally[k]
    x = allx[k]
    if x>4 and x<2043 and y>4 and y<2043: 
        annulus_apertures = CircularAnnulus([(x,y)], r_in=3., r_out=10.)
        phot_table1 = aperture_photometry(flat1clipped, annulus_apertures)
        phot_table2 = aperture_photometry(flat1unity, annulus_apertures)
        bkg_mean = phot_table1['aperture_sum'] / phot_table2['aperture_sum'] 

        #If adjacent pixels have low qe, set those to the background for this test
        neigh1 = flat1[y+1,x]
        neigh2 = flat1[y-1,x]
        neigh3 = flat1[y,x+1]
        neigh4 = flat1[y,x-1]
        if neigh1<thresh_lowqe:
            neigh1 = bkg_mean
        if neigh2<thresh_lowqe:
            neigh2 = bkg_mean
        if neigh3<thresh_lowqe:
            neigh3 = bkg_mean
        if neigh4<thresh_lowqe:
            neigh4 = bkg_mean
        fluxneigh1 = (neigh1+neigh2+neigh3+neigh4)/4.0

        neighexcess = fluxneigh1/bkg_mean[0]
        if neighexcess>=thresh_nearopen:
            adjopen[y+1,x+1] = 1
            adjopen[y-1,x-1] = 1
            adjopen[y-1,x+1] = 1
            adjopen[y+1,x-1] = 1
        if neighexcess>=thresh_adjopen:
            openpixel[y,x] = 1
            adjopen[y+1,x] = 1
            adjopen[y-1,x] = 1
            adjopen[y,x+1] = 1
            adjopen[y,x-1] = 1
        else:
            if flat1[y,x]<=thresh_dead:
                dead[y,x] = 1
            else:    
                lowqe[y,x] = 1

#For dead pixels use the NO_SAT_CHECK flag as well - actually this does not go in the bad pixel mask file
#Dead pixels do not migrate charge to neighbors. This will stop neighbors being flagged as saturated by the pipeline.               
#w = np.where(dead==1)
#nosatcheck[w] = 1
                
#Reset any adjacent to open that are actually RC, low qe in clusters, or unreliable bias that are also low qe
w = np.where((rc==1)|(openpixel==1)|(dead==1)|(lowqe==1))
adjopen[w] = 0
unrelbias[w] = 0

#Reset from these categories all RC pixels
w = np.where(rc==1)
openpixel[w] = 0
dead[w] = 0
lowqe[w] = 0

#==============================================================
#Clean up bad pixel classifications and fill bpm arrays

#Reset "other bad" category if already in a different category
w = np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(telegraph==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(unreldarkshort==1)|(unrelslopeshort==1))
otherbadshort[w] = 0
w = np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(telegraph==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(unreldarklong==1)|(unrelslopelong==1))
otherbadlong[w] = 0
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(telegraph==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(unreldarkminimal==1)|(unrelslopeminimal==1))
otherbadminimal[w] = 0

#Set DO_NOT_USE short and long for some bad pixel types

w = np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(badrefpixelshort==1)|(unreldarkshort==1)|(unrelslopeshort==1)|(otherbadshort==1))
donotuseshort[w] = 1
w = np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(badrefpixellong==1)|(unreldarklong==1)|(unrelslopelong==1)|(otherbadlong==1))
donotuselong[w] = 1
w=np.where((dead==1)|(hot==1)|(lowqe==1)|(rc==1)|(unrelbias==1)|(unrelflat==1)|(openpixel==1)|(adjopen==1)|(badrefpixelminimal==1)|(unreldarkminimal==1)|(unrelslopeminimal==1)|(otherbadminimal==1))
donotuseminimal[w] = 1

#Only mark nonlinear as DO_NOT_USE if includelinearity==True
#Deprecated
#if includelinearity==True:
#    donotuseshort[np.where(nonlinear==1)]   = 1
#    donotuselong[np.where(nonlinear==1)]    = 1
#    donotuseminimal[np.where(nonlinear==1)] = 1
#    logging.info('Setting nonlinear pixels as DO_NOT_USE')

#Set up flagtable and primary header & DQ_DEF tables of mask reference files
maskhduprimary,maskhdudqdeflong,flagtable = setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise_longonly,thresh_cdsnoise,author,pedigree,useafter,fpa_temp,description_mask,history_mask)
maskhduprimary,maskhdudqdefshort,flagtable = setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise,thresh_cdsnoise_shortonly,author,pedigree,useafter,fpa_temp,description_mask,history_mask)
maskhduprimary,maskhdudqdefminimal,flagtable = setupmaskhdu(thresh_lowqe,thresh_dead,thresh_adjopen,thresh_nearopen,thresh_hot,thresh_warm,thresh_cold,thresh_bias,thresh_bias_sigma,thresh_slopenoise,thresh_cdsnoise,author,pedigree,useafter,fpa_temp,description_mask,history_mask)

#Assign bad pixel bit values to bpm
bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'REFERENCE_PIXEL')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(refpixel>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'DEAD')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(dead>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'HOT')][0]
flagarray=np.ones(arrshape, dtype=int)*bitvalue
w = np.where(hot>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'WARM')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(warm>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'LOW_QE')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(lowqe>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'RC')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(rc>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'TELEGRAPH')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w=np.where(telegraph>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

#bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'NONLINEAR')][0]
#flagarray = np.ones(arrshape, dtype=int)*bitvalue
#w = np.where(nonlinear>0)
#bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

#bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'NO_SAT_CHECK')][0]
#flagarray = np.ones(arrshape, dtype=int)*bitvalue
#w = np.where(nosatcheck>0)
#bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_BIAS')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(unrelbias>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

#bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_FLAT')][0]
#flagarray = np.ones(arrshape, dtype=int)*bitvalue
#w = np.where(unrelflat>0)
#bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'OPEN')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(openpixel>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'ADJ_OPEN')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(adjopen>0)
bpm[w] = np.bitwise_or(bpm[w],flagarray[w])

#Assign bad pixel bit values to bpm for attributes that are different for short, long and minimal darks
bpmlong = deepcopy(bpm)
bpmshort = deepcopy(bpm)
bpmminimal = deepcopy(bpm)

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_DARK')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(unreldarklong>0)
bpmlong[w] = np.bitwise_or(bpmlong[w],flagarray[w])
w = np.where(unreldarkshort>0)
bpmshort[w] = np.bitwise_or(bpmshort[w],flagarray[w])
w = np.where(unreldarkminimal>0)
bpmminimal[w] = np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_SLOPE')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(unrelslopelong>0)
bpmlong[w] = np.bitwise_or(bpmlong[w],flagarray[w])
w = np.where(unrelslopeshort>0)
bpmshort[w] = np.bitwise_or(bpmshort[w],flagarray[w])
w = np.where(unrelslopeminimal>0)
bpmminimal[w] = np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'DO_NOT_USE')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(donotuselong>0)
bpmlong[w] = np.bitwise_or(bpmlong[w],flagarray[w])
w = np.where(donotuseshort>0)
bpmshort[w] = np.bitwise_or(bpmshort[w],flagarray[w])
w = np.where(donotuseminimal>0)
bpmminimal[w] = np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'OTHER_BAD_PIXEL')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where(otherbadlong>0)
bpmlong[w] = np.bitwise_or(bpmlong[w],flagarray[w])
w = np.where(otherbadshort>0)
bpmshort[w] = np.bitwise_or(bpmshort[w],flagarray[w])
w = np.where(otherbadminimal>0)
bpmminimal[w] = np.bitwise_or(bpmminimal[w],flagarray[w])

bitvalue = flagtable['Value'][np.where(flagtable['Name']==b'BAD_REF_PIXEL')][0]
flagarray = np.ones(arrshape, dtype=int)*bitvalue
w = np.where((donotuselong>0)&(refpixel>0))
badrefpixellong[w] = 1
bpmlong[w] = np.bitwise_or(bpmlong[w],flagarray[w])
w = np.where((donotuseshort>0)&(refpixel>0))
badrefpixelshort[w] = 1
bpmshort[w] = np.bitwise_or(bpmshort[w],flagarray[w])
w = np.where((donotuseminimal>0)&(refpixel>0))
badrefpixelminimal[w] = 1
bpmminimal[w] = np.bitwise_or(bpmminimal[w],flagarray[w])

#=====================================================================
#Write final readnoise and mask reference files

#Set up primary header and SCI extensions of readnoise reference file
readnoisehduprimary = setupreadnoisehdu(author,pedigree,useafter,readpattern,fpa_temp,description_readnoise,history_readnoise)
readnoisehdusci = fits.ImageHDU(readnoise,name='SCI')
#Write the readnoise reference file
readnoisehdulist = fits.HDUList([readnoisehduprimary,readnoisehdusci])
readnoisehdulist['SCI'].header['BUNIT']='DN'
readnoisehdulist.writeto(readnoisefile, overwrite=True)

#Set up  DQ extensions of mask reference files
maskhdudqlong = fits.ImageHDU(bpmlong,name = 'DQ')
maskhdudqshort = fits.ImageHDU(bpmshort,name='DQ')
maskhdudqminimal = fits.ImageHDU(bpmminimal,name='DQ')
#Write the mask reference files
maskhdulist = fits.HDUList([maskhduprimary,maskhdudqlong,maskhdudqdeflong])
maskhdulist.writeto(longbpmfile, overwrite=True)
maskhdulist = fits.HDUList([maskhduprimary,maskhdudqshort,maskhdudqdefshort])
maskhdulist.writeto(shortbpmfile, overwrite=True)
maskhdulist = fits.HDUList([maskhduprimary,maskhdudqminimal,maskhdudqdefminimal])
maskhdulist.writeto(minimalbpmfile, overwrite=True)

logging.info('')
logging.info('Wrote reference files:')
logging.info(readnoisefile)
logging.info(longbpmfile)
logging.info(shortbpmfile)
logging.info(minimalbpmfile)

print ('Finished writing reference files')

#=====================================================================
#Log number of bad pixels per type

logging.info('')
logging.info('Number and percentage of bad pixels of each type:')
w = np.where(donotuselong==1)
logging.info('donotuselong; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(donotuseshort==1)
logging.info('donotuseshort; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(donotuseminimal==1)
logging.info('donotuseminimal; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(dead==1)
logging.info('dead; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(hot==1)
logging.info('hot; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(warm==1)
logging.info('warm; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(lowqe==1)
logging.info('lowqe; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(rc==1)
logging.info('rc; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(telegraph==1)
logging.info('telegraph; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
#w = np.where(nonlinear==1)
#logging.info('nonlinear; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
#w = np.where(nosatcheck==1)
#logging.info('nosatcheck; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(badrefpixellong==1)
logging.info('badrefpixellong; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*8.0+2040.0*8.0))))
w = np.where(badrefpixelshort==1)
logging.info('badrefpixelshort; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*8.0+2040.0*8.0))))
w = np.where(badrefpixelminimal==1)
logging.info('badrefpixelminimal; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*8.0+2040.0*8.0))))
w = np.where(unrelbias==1)
logging.info('unrelbias; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(unreldarklong==1)
logging.info('unreldarklong; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(unreldarkshort==1)
logging.info('unreldarkshort; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(unreldarkminimal==1)
logging.info('unreldarkminimal; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(unrelslopelong==1)
logging.info('unrelslopelong; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(unrelslopeshort==1)
logging.info('unrelslopeshort; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(unrelslopeminimal==1)
logging.info('unrelslopeminimal; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
#w = np.where(unrelflat==1)
#logging.info('unrelflat; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(openpixel==1)
logging.info('openpixel; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(adjopen==1)
logging.info('adjopen; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(otherbadlong==1)
logging.info('otherbadlong; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(otherbadshort==1)
logging.info('otherbadshort; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))
w = np.where(otherbadminimal==1)
logging.info('otherbadminimal; N = {}, %= {}'.format(len(w[0]),(100.0*len(w[0])/(2048.0*2048.0))))

#====================================================================================================
#Optionally, output a separate file for each type of bad pixel (useful for inspection and correlation) 
if outputseparatefiles==True:
    if not os.path.exists(typebpmdir):
        os.makedirs(typebpmdir)
    donotuselongfile = typebpmfile+'_donotuse_long.fits'
    fits.writeto(donotuselongfile,donotuselong,overwrite=True)
    donotuseshortfile = typebpmfile+'_donotuse_short.fits'
    fits.writeto(donotuseshortfile,donotuseshort,overwrite=True)
    donotuseminimalfile = typebpmfile+'_donotuse_minimal.fits'
    fits.writeto(donotuseminimalfile,donotuseminimal,overwrite=True)
    deadfile = typebpmfile+'_dead.fits'
    fits.writeto(deadfile,dead,overwrite=True)
    hotfile = typebpmfile+'_hot.fits'
    fits.writeto(hotfile,hot,overwrite=True)
    warmfile = typebpmfile+'_warm.fits'
    fits.writeto(warmfile,warm,overwrite=True)
    lowqefile = typebpmfile+'_lowqe.fits'
    fits.writeto(lowqefile,lowqe,overwrite=True)
    rcfile = typebpmfile+'_rc.fits'
    fits.writeto(rcfile,rc,overwrite=True)
    telegraphfile = typebpmfile+'_telegraph.fits'
    fits.writeto(telegraphfile,telegraph,overwrite=True)
    #nonlinearfile = typebpmfile+'_nonlinear.fits'
    #fits.writeto(nonlinearfile,nonlinear,overwrite=True)
    #nosatcheckfile = typebpmfile+'_nosatcheck.fits'
    #fits.writeto(nosatcheckfile,nosatcheck,overwrite=True)
    badrefpixellongfile = typebpmfile+'_badrefpixel_long.fits'
    fits.writeto(badrefpixellongfile,badrefpixellong,overwrite=True)
    badrefpixelshortfile = typebpmfile+'_badrefpixel_short.fits'
    fits.writeto(badrefpixelshortfile,badrefpixelshort,overwrite=True)
    badrefpixelminimalfile = typebpmfile+'_badrefpixel_minimal.fits'
    fits.writeto(badrefpixelminimalfile,badrefpixelminimal,overwrite=True)
    unrelbiasfile = typebpmfile+'_unrelbias.fits'
    fits.writeto(unrelbiasfile,unrelbias,overwrite=True)
    unreldarklongfile = typebpmfile+'_unreldark_long.fits'
    fits.writeto(unreldarklongfile,unreldarklong,overwrite=True)
    unreldarkshortfile = typebpmfile+'_unreldark_short.fits'
    fits.writeto(unreldarkshortfile,unreldarkshort,overwrite=True)
    unreldarkminimalfile = typebpmfile+'_unreldark_minimal.fits'
    fits.writeto(unreldarkminimalfile,unreldarkminimal,overwrite=True)
    unrelslopelongfile = typebpmfile+'_unrelslope_long.fits'
    fits.writeto(unrelslopelongfile,unrelslopelong,overwrite=True)
    unrelslopeshortfile = typebpmfile+'_unrelslope_short.fits'
    fits.writeto(unrelslopeshortfile,unrelslopeshort,overwrite=True)
    unrelslopeminimalfile = typebpmfile+'_unrelslope_minimal.fits'
    fits.writeto(unrelslopeminimalfile,unrelslopeminimal,overwrite=True)
    #unrelflatfile = typebpmfile+'_unrelflat.fits'
    #fits.writeto(unrelflatfile,unrelflat,overwrite=True)
    openpixelfile = typebpmfile+'_openpixel.fits'
    fits.writeto(openpixelfile,openpixel,overwrite=True)
    adjopenfile = typebpmfile+'_adjopen.fits'
    fits.writeto(adjopenfile,adjopen,overwrite=True)
    otherbadlongfile = typebpmfile+'_otherbad_long.fits'
    fits.writeto(otherbadlongfile,otherbadlong,overwrite=True)
    otherbadshortfile = typebpmfile+'_otherbad_short.fits'
    fits.writeto(otherbadshortfile,otherbadshort,overwrite=True)
    otherbadminimalfile = typebpmfile+'_otherbad_minimal.fits'
    fits.writeto(otherbadminimalfile,otherbadminimal,overwrite=True)
    refpixelfile = typebpmfile+'_refpixel.fits'
    fits.writeto(refpixelfile,refpixel,overwrite=True)
