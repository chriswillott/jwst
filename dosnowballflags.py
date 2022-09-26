#!/usr/bin/env python

import numpy as np
import os
from astropy.io import fits
from jwst.datamodels import dqflags
from photutils.segmentation import detect_sources
from photutils.segmentation import SourceCatalog
from copy import deepcopy 
from skimage.draw import disk


def snowballflags(jumpdirfile,filtername,npixfind,satpixradius,halofactorradius,imagingmode):
    """
    Flag pixels in snowballs - expand saturated ring and diffuse halo jump ring.
    The GROUPDQ array will be flagged with the SATURATED and JUMP_DET flags.
    Saturating snowballs early in short ramps can have unflagged central pixels that jump in the previous group.
    This is called after the regular jump step.
    The output file is overwritten.
    You need a working installation of WebbPSF.
    Requires checkifstar.py if imagingmode is True

    Parameters:
    jumpdirfile - path to the input file, which has jump step applied 
    filtername  - used for comparing to a WebbPSF star 
    npixfind - number of connected pixels to find snowballs
    satpixradius - how many extra pixels to flag as saturated to account for slow charge migration near the saturated core - default 2
    halofactorradius - factor to increase radius of whole snowball for jump flagging - default 2.0  
    imagingmode - boolean for whether imaging or spectroscopy. For imaging mode checks to see if the detected object is a star and then does not expand DQ arrays
    """
    with fits.open(jumpdirfile) as hdulist:
        sci = hdulist['SCI'].data     
        pdq = hdulist['PIXELDQ'].data    
        gdq = hdulist['GROUPDQ'].data     
        header = hdulist[0].header
        ins = header['INSTRUME'].lower()
        nint = header['NINTS']
        ngroup = header['NGROUPS']
        grouparray=np.arange(ngroup)+1   
       
        #Set up webbpsf psfs for checking if a star
        if ins=='niriss':
            pixscale=0.0656
            cutsize=37
            radmin = 9
            radmax = 16
            spikeratio = 1.4
        elif ins=='nircam':
            channel = header['CHANNEL']
            if channel == 'SHORT': 
                pixscale = 0.033
                cutsize = 74
                radmin = 18
                radmax = 32
                spikeratio = 1.4
            else:     
                pixscale = 0.66
                cutsize = 37
                radmin = 9
                radmax = 16
                spikeratio = 1.4
        elif ins=='nirspec':
            #Note these parameters only used for imaging mode so dummy values for NIRSpec
            pixscale = 0.10
            cutsize = 25
            radmin = 6
            radmax = 12
            spikeratio = 1.4
                
        #Make the WebbPSF mask (will not repeat the actual WebbPSF call if the file already exists)
        if imagingmode == True:
            from checkifstar import checkif, makewebbpsfmask
            print ('Running makewebbpsfmask',ins,filtername,pixscale,cutsize,radmin,radmax)
            webbpsfcutoutmask = makewebbpsfmask(ins,filtername,pixscale,cutsize,radmin,radmax)    

        #iterate over integrations
        for h in range(nint):
            
            ctsnow=0
            #Skip first group because no jumps there
            for j in range(1,ngroup):
                #Find pixels in this group with jump detected and/or saturation detected
                jumps = np.zeros((2048,2048),dtype='uint8') 
                sat   = np.zeros((2048,2048),dtype='uint8') 
                jumpsorsat = np.zeros((2048,2048),dtype='uint8') 
                scithisgroup = np.squeeze(sci[h,j,:,:])          
                dqthisgroup = np.squeeze(gdq[h,j,:,:])
                i_yy,i_xx, = np.where(np.bitwise_and(dqthisgroup, dqflags.group['JUMP_DET'])  != 0)
                jumps[i_yy,i_xx] = 1
                jumpsorsat[i_yy,i_xx] = 1
                i_yy,i_xx, = np.where(np.bitwise_and(dqthisgroup, dqflags.group['SATURATED']) != 0)
                sat[i_yy,i_xx] = 1
                jumpsorsat[i_yy,i_xx] = 1
                
                #Set some low threshold for finding sources in noiseless DQ array
                threshsigma = 3.0
                bkg = 0.0
                stddev = 0.00007
                photthreshold = bkg + (threshsigma * stddev)
                
                #Run initial find on jumps or saturated because some short ramps do not have a jump in the regions that saturate
                segm_detect = detect_sources(jumpsorsat, photthreshold, npixels=npixfind)
                segimage = segm_detect.data.astype(np.uint32)
                if np.max(segimage)>0:
                    segmcat = SourceCatalog(jumps, segm_detect)  
                    segmtbl = segmcat.to_table()
                    ctsnowballs = segmtbl['xcentroid'][:].size
                    #print (j,ctsnowballs,segmtbl)
                    #Iterate over each possible snowball
                    for k in range(ctsnowballs):
                        #If low eccentricity proceed, otherwise remove source from segmentation image
                        #Use both eccentricity and segmentation box axis since not always consistent, e.g. for merged jumps
                        segboxaxisratio = np.abs((segmtbl['bbox_xmax'][k]-segmtbl['bbox_xmin'][k])/(segmtbl['bbox_ymax'][k]-segmtbl['bbox_ymin'][k]))
                        if segboxaxisratio<1.0:
                            segboxaxisratio = np.abs((segmtbl['bbox_ymax'][k]-segmtbl['bbox_ymin'][k])/(segmtbl['bbox_xmax'][k]-segmtbl['bbox_xmin'][k]))
                        if ((segmtbl['eccentricity'][k]<0.6)&(segboxaxisratio<1.5)):
                            #print (j,k+1,segmtbl['xcentroid'][k],segmtbl['ycentroid'][k],'eccen=',segmtbl['eccentricity'][k],segmtbl['bbox_ymin'][k],segmtbl['bbox_ymax'][k],segmtbl['bbox_xmin'][k],segmtbl['bbox_xmax'][k],segboxaxisratio)
                            #Check if a star by running the checkifstar.py code on the relevant group of the jump cube sci array masking out bad pixels inc jump and saturated pixels
                            #First cutout should be same size as WebbPSF PSF
                            if imagingmode == True:
                                xlo = int(segmtbl['xcentroid'][k]-(cutsize-1)/2)
                                xhi = xlo+cutsize
                                ylo = int(segmtbl['ycentroid'][k]-(cutsize-1)/2)
                                yhi = ylo+cutsize    

                                scicutout   = deepcopy(scithisgroup[ylo:yhi,xlo:xhi])
                                pdqcutout   = deepcopy(pdq[ylo:yhi,xlo:xhi])
                                jumpscutout = jumps[ylo:yhi,xlo:xhi]
                                satcutout   = sat[ylo:yhi,xlo:xhi]
                                pdqcutout[np.where(jumpscutout>0)] = 1
                                pdqcutout[np.where(satcutout>0)] = 1
                                #Run the check to see if this is a saturated star rather than a snowball
                                isstar = checkif(scicutout,pdqcutout,webbpsfcutoutmask,radmin,radmax,spikeratio)
                            else:
                                isstar = False

                            if isstar == False:
                                jumpscutout = jumps[int(segmtbl['bbox_ymin'][k]):int(segmtbl['bbox_ymax'][k]),int(segmtbl['bbox_xmin'][k]):int(segmtbl['bbox_xmax'][k])]
                                satcutout   = sat[int(segmtbl['bbox_ymin'][k]):int(segmtbl['bbox_ymax'][k]),int(segmtbl['bbox_xmin'][k]):int(segmtbl['bbox_xmax'][k])]
                                jumpsorsatcutout = jumpsorsat[int(segmtbl['bbox_ymin'][k]):int(segmtbl['bbox_ymax'][k]),int(segmtbl['bbox_xmin'][k]):int(segmtbl['bbox_xmax'][k])]
                                #Triple box size for increased area to flag further out
                                bigoffsetx = int((segmtbl['bbox_xmax'][k]-int(segmtbl['bbox_xmin'][k])))
                                bigoffsety = int((segmtbl['bbox_ymax'][k]-int(segmtbl['bbox_ymin'][k])))
                                bigsizex = jumpscutout.shape[1]+2*bigoffsetx
                                bigsizey = jumpscutout.shape[0]+2*bigoffsety
                                jumpsbigcutout = np.zeros((bigsizey,bigsizex),dtype=np.uint8)
                                jumpsbigcutout[bigoffsety:(bigoffsety+jumpscutout.shape[0]),bigoffsetx:(bigoffsetx+jumpscutout.shape[1])] = jumpscutout
                                satbigcutout = np.zeros((bigsizey,bigsizex),dtype=np.uint8)
                                satbigcutout[bigoffsety:(bigoffsety+jumpscutout.shape[0]),bigoffsetx:(bigoffsetx+jumpscutout.shape[1])] = satcutout

                                #For jumps assume round and use all jump or saturated pixels to get area
                                numjumporsat = jumpsorsatcutout[np.where(jumpsorsatcutout>0)].size
                                radiusjumporsat = (numjumporsat/3.14159)**0.5             
                                radius = int(halofactorradius*radiusjumporsat)
                                rr, cc = disk((bigsizey/2-0.5,bigsizex/2-0.5), radius)
                                jumpsbigcutout[rr, cc] = 4

                                #For saturation assume round and use saturated pixels to get area
                                numsat = satcutout[np.where(satcutout>0)].size
                                radiussat = (numsat/3.14159)**0.5             
                                radius = int(radiussat+satpixradius)
                                rr, cc = disk((bigsizey/2-0.5,bigsizex/2-0.5), radius)
                                satbigcutout[rr, cc] = 2

                                xlo = int(segmtbl['bbox_xmin'][k])-bigoffsetx
                                xhi = xlo+bigsizex
                                ylo = int(segmtbl['bbox_ymin'][k])-bigoffsety
                                yhi = ylo+bigsizey

                                #Update pixels in GROUPDQ array for halo
                                i_yy,i_xx, = np.where(jumpsbigcutout>0)
                                i_yy+=ylo
                                i_xx+=xlo
                                numpix = len(i_xx)
                                for l in range(numpix):
                                    if ((i_xx[l]>3) & (i_xx[l]<2044) & (i_yy[l]>3) & (i_yy[l]<2044)):
                                        gdq[h,j,i_yy[l],i_xx[l]] = np.bitwise_or(gdq[h,j,i_yy[l],i_xx[l]],dqflags.group['JUMP_DET'])

                                #Update pixels in GROUPDQ array for saturated core
                                i_yy,i_xx, = np.where(satbigcutout>0)
                                i_yy+=ylo
                                i_xx+=xlo
                                numpix = len(i_xx)
                                for l in range(numpix):
                                    if ((i_xx[l]>3) & (i_xx[l]<2044) & (i_yy[l]>3) & (i_yy[l]<2044)):
                                        gdq[h,j,i_yy[l],i_xx[l]] = np.bitwise_or(gdq[h,j,i_yy[l],i_xx[l]],dqflags.group['SATURATED'])

                                #if the snowball happened in the third group, flag the second group similarly in case first effects happened there and not enough data for good ramps up to there anyway.
                                if j==2:
                                    i_yy,i_xx, = np.where(jumpsbigcutout>0)
                                    i_yy+=ylo
                                    i_xx+=xlo                    
                                    numpix = len(i_xx)
                                    for l in range(numpix):
                                        if ((i_xx[l]>3) & (i_xx[l]<2044) & (i_yy[l]>3) & (i_yy[l]<2044)):
                                            gdq[h,j-1,i_yy[l],i_xx[l]] = np.bitwise_or(gdq[h,j-1,i_yy[l],i_xx[l]],dqflags.group['JUMP_DET'])

                                    i_yy,i_xx, = np.where(satbigcutout>0)
                                    i_yy+=ylo
                                    i_xx+=xlo
                                    numpix = len(i_xx)
                                    for l in range(numpix):
                                        if ((i_xx[l]>3) & (i_xx[l]<2044) & (i_yy[l]>3) & (i_yy[l]<2044)):
                                            gdq[h,j-1,i_yy[l],i_xx[l]] = np.bitwise_or(gdq[h,j-1,i_yy[l],i_xx[l]],dqflags.group['SATURATED'])

                                ctsnow+=1   
                                
            #Any pixel flagged as saturated in a group must be flagged as saturated in all subsequent groups  
            for j in range(ngroup):
                #Find pixels in this group with saturation detected
                sat   = np.zeros((2048,2048),dtype='uint8') 
                dqthisgroup = np.squeeze(gdq[h,j,:,:])
                i_yy,i_xx, = np.where(np.bitwise_and(dqthisgroup, dqflags.group['SATURATED']) != 0)
                numpix = len(i_xx)
                for l in range(numpix):
                    gdq[h,j:,i_yy[l],i_xx[l]] = np.bitwise_or(gdq[h,j:,i_yy[l],i_xx[l]],dqflags.group['SATURATED'])
                    
            header['HISTORY'] = 'Corrected {} snowballs in integration {}'.format(ctsnow,(h+1))
            
            
        hdulist['GROUPDQ'].data = gdq 
        header.set('SNOWCORR', 'COMPLETE', 'dosnowballflags.py DQ flagging applied')
        hdulist[0].header = header
        #uncomment below to output ramp in a different file
        #snowfile = jumpdirfile.replace('.fits','_snow.fits')
        snowfile = jumpdirfile
        hdulist.writeto(snowfile,overwrite=True)  

#Run directly for testing
direct=False
if direct:
    jumpdirfile = './jw01345001001_02201_00001_nrca1_jump.fits'
    imagingmode = True
    filtername = 'F115W'
    npixfind = 50
    satpixradius=3
    halofactorradius=2
    
    snowballflags(jumpdirfile,filtername,npixfind,satpixradius,halofactorradius,imagingmode)
