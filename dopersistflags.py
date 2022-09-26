#!/usr/bin/env python

import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
from jwst.datamodels import dqflags
from jwst.refpix.irs2_subtract_reference import make_irs2_mask
from datetime import datetime
import logging

def persistflags(prevrawdirfile,rawdirfile,jumpdirfile,countlimit,timeconstant):
    """
    Flag pixels that reached a very high number of counts (countlimit) in the previous integration
    The GROUPDQ array will be flagged with the JUMP_DET flag (this is called after the jump step and snowball flagging so won't interfere with that) 
    Only groups within timeconstant after end of previous integration are flagged
    The input file is overwritten
    
    Parameters:
    prevrawdirfile - path to the raw exposure taken before this one, can set equal to ' ' if there is not a previous exposure in the dataset  
    rawdirfile - path to this raw exposure
    jumpdirfile - path to the input ramp file, which has jump step and snowball flagging applied
    countlimit  - count level in raw data in previous integration above which to flag persistence in following integration - default 50000
    timeconstant - how long after end of previous integration to flag groups  
    """
    with fits.open(jumpdirfile) as hdulist:
        gdq = hdulist['GROUPDQ'].data     
        header = hdulist[0].header
        detector = header['DETECTOR']
        readpatt = header['READPATT']
        nint = header['NINTS']
        tgroup = header['TGROUP']
        ngroup = header['NGROUPS']
        grouparray=np.arange(ngroup)+1  
        inttimes = Table.read(rawdirfile, hdu='INT_TIMES')

        #iterate over integrations
        for h in range(nint):
            #print ('working on int',(h+1))
            previntend = 0.0
        
            if h==0:            
                #for first integration must get raw data from last integration of previous exposure if it exists
                if os.path.exists(prevrawdirfile):
                    rawheader = fits.getheader(prevrawdirfile)
                    rawdata = fits.getdata(prevrawdirfile)
                    #If data is NIRSpec IRS2 need to removed interleaved reference pixels
                    if readpatt=='NRSIRS2RAPID' or readpatt=='NRSIRS2':
                        rawdata = reshapeirs2(rawdata,detector,header)
                    maxrawdata = np.max(np.squeeze(rawdata[-1,:,:,:]),axis=0)
                    intstart = inttimes['int_start_MJD_UTC'][0]
                    previnttimes = Table.read(prevrawdirfile, hdu='INT_TIMES')
                    previntend = previnttimes['int_end_MJD_UTC'][-1]
            else:            
                #for later integrations must get raw data from same file
                rawheader = fits.getheader(rawdirfile)
                rawdata = fits.getdata(rawdirfile)
                #If data is NIRSpec IRS2 need to removed interleaved reference pixels
                if readpatt=='NRSIRS2RAPID' or readpatt=='NRSIRS2':
                    rawdata = reshapeirs2(rawdata,detector,header)
                maxrawdata = np.max(np.squeeze(rawdata[(h-1),:,:,:]),axis=0)
                intstart = inttimes['int_start_MJD_UTC'][h]
                previntend = inttimes['int_end_MJD_UTC'][(h-1)]

            #skip the update if there was no previous integration in the raw file set    
            if previntend>0:    
                satindices = np.where(maxrawdata>countlimit)
                i_yy,i_xx, = np.where(maxrawdata>countlimit)
                numsat = maxrawdata[satindices].size
                #print ('number of pixels above countlimit ',numsat)

                mjdintegration = intstart+((grouparray+0.5)*tgroup/(24*3600.0))
                timesinceend = (mjdintegration - previntend)*(24*3600.0)

                for k in range(1,ngroup):
                    if timesinceend[k] < timeconstant:
                        gdq[h,k,i_yy,i_xx] = np.bitwise_or(gdq[h,k,i_yy,i_xx],dqflags.group['JUMP_DET'])     
                
        #write out results
        outfile = jumpdirfile
        #uncomment below to output ramp in a different file
        #outfile = jumpdirfile.replace('.fits','_persist.fits')
        hdulist['GROUPDQ'].data = gdq     
        hdulist.writeto(outfile, overwrite=True)

        
def reshapeirs2(rawdata,detector,header):
    """
    Take in a raw NIRSpec IRS2 data array and return an array containing only the non-reference pixels.
    """
    
    #switch from DMS to detector orientation
    if detector == "NRS1":
        rawdata = np.swapaxes(rawdata, 2, 3)
    elif detector == "NRS2":
        rawdata = np.swapaxes(rawdata, 2, 3)[:, :, ::-1, ::-1]
    ny = rawdata.shape[-2]                
    nx = rawdata.shape[-1]
    scipix_n = header['NRS_NORM']
    refpix_r = header['NRS_REF']
    #use jwst pipeline function to make a 1D mask of the non-reference pixels
    irs2_mask = make_irs2_mask(nx, ny, scipix_n, refpix_r)
    rawdata = rawdata[:, :, :, irs2_mask]
    #switch back from detector to DMS orientation 
    if detector == "NRS1":
        rawdata = np.swapaxes(rawdata, 2, 3)
    elif detector == "NRS2":
        rawdata = np.swapaxes(rawdata[:, :, ::-1, ::-1], 2, 3)
                    
    return rawdata


#Run directly for testing
direct=False
if direct:
    prevrawdirfile = 'jw01222002001_03104_00002_nrs1_uncal.fits'
    rawdirfile =    'jw01222002001_03104_00003_nrs1_uncal.fits'
    jumpdirfile =  'jw01222002001_03104_00003_nrs1_jump.fits'
    countlimit=50000
    timeconstant=1000.0
    persistflags(prevrawdirfile,rawdirfile,jumpdirfile,countlimit,timeconstant)
