#!/usr/bin/env python

#Generate NIRISS WFSS grism flat field reference files by correcting the imaging flat with data from grism flats.
#Uses jwst and jwst_reffiles python packages
#Takes in direct and dispersed NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror, including the coronagraphic spots.
#Inputs:
#Place all imaging flat field reference files in one sub-directory per filter under the input directory indirimages, e.g. ./imageflatreffiles/F150W/. These should be normalized to a sigma-clipped average value of one as per the CALWEBB_IMAGE2 definition.
#Place one GR150C and one GR150R grism flat field slope_rate.fits files in a sub-directory per filter (for F115W and F150W only) under the input directory indirgrism, e.g. ./grismflatinputs/F150W/
#The POM feature 'detect' direct image and grism flats are usually F115W because this is best to detect the features (true for CV3 OSIM with very red source, will not be true for astrophysical source).
#The POM feature 'measure' direct image flat is usually the output filter due to the wavelength-dependence of flux loss.  
#The four coronagraphic spots have too low S/N in the POM transmission image so will replace with regions from locally normalized  F150W grism flats.
#A bad pixel mask reference file is required so that interpolation is applied across bad pixels.
#The outline of the rotated POM in the oversized field determined from the F200W dispersed flats. Must contain keywords OFFSETX & OFFSETY. 
#Outputs:
#Final grism flat field images will be output to one sub-directory per filter under the output directory, e.g. ./grismflatreffiles/F150W/
#A POM transmission file is output in the same directory. This file is used to model sources in WFSS images.
#The parameter ngrow (default=1) is the number of pixels by which to grow the segmentation map of each detected source.

#Usage:
#makenirissgrismflats.py  --indirimage='./imageflatreffiles' --indirgrism='./grismflatinputs'  --mask='/Users/willottc/niriss/detectors/willott_reference_files/nocrs/jwst_niriss_cv3_38k_nocrs_bpm_minimal.fits' --pomoutline='./plots/pomoutlinecv3flats_f200w.fits' --outdir='./grismflatreffiles'  --ngrow=1

import numpy as np
from scipy import ndimage
import optparse, sys
import os
import astropy.io.fits as fits
from astropy.stats import SigmaClip,sigma_clip,sigma_clipped_stats
from copy import deepcopy
from photutils import detect_threshold,detect_sources,Background2D, MedianBackground,source_properties, CircularAperture, CircularAnnulus
import natsort
from datetime import datetime
from jwst.datamodels import FlatModel, util, dqflags
from jwst_reffiles.bad_pixel_mask import bad_pixel_mask

# Command line options
op = optparse.OptionParser()
op.add_option("--indirimage")
op.add_option("--indirgrism")
op.add_option("--mask")
op.add_option("--pomoutline")
op.add_option("--outdir")
op.add_option("--ngrow")

o, a = op.parse_args()
if a:
    print (sys.stderr, "unrecognized option: ",a)
    sys.exit(-1)

indirimage=o.indirimage
indirgrism=o.indirgrism
maskfile=o.mask
pomoutlinefile=o.pomoutline
outdir=o.outdir
ngrow=o.ngrow
if ngrow==None:
    ngrow=1

if not os.path.exists(outdir):
    os.makedirs(outdir)

def save_final_map(datamap, dq, err, dqdef, instrument, detector, hdulist, filterdir, grism, files,
                       author, description, pedigree,useafter, fpatemp, history_text, pomoutlineoffsetx, pomoutlineoffsety, ngrow, outfile):
    """Save a flat field or POM transmission map into a CRDS-formatted reference file
    Parameters
    ----------
    datamap : numpy.ndarray
        2D flat-field array
    dq : numpy.ndarray
        2D flat-field DQ array
    err : numpy.ndarray
        2D flat-field error array
    dqdef : numpy.ndarray
        binary table of DQ definitions
    instrument : str
        Name of instrument associated with the flat-field array
    detector : str
        Name of detector associated with the flat-field array
    hdulist : astropy.fits.HDUList
        HDUList containing "extra" fits keywords
    filterdir : str
        filter of flat file
    grism : str
        grism of flat file
    files : list
        List of files used to create reference file
    author : str
        Author of the reference file
    description : str
        CRDS description to use in the final reference file
    pedigree : str
        CRDS pedigree to use in the final reference file
    useafter : str
        CRDS useafter string for the reference file
    history_text : list
        List of strings to add as HISTORY entries to the reference file
    outfile : str
        Name of the output reference file
    """
    yd, xd = datamap.shape

    # For now use FlatModel for the POM transmission as well so don't have to define a new model.
    # Initialize the FlatModel using the hdu_list, so the new keywords will
    # be populated
    if 'flat' in description:
        model = FlatModel(hdulist)
        model.meta.reftype = 'FLAT'
    elif 'transmission' in description:
        model = FlatModel(hdulist)
        model.meta.reftype = 'TRANSMISSION'
    model.data = datamap
    model.dq = dq
    model.err = err
    model.dq_def = dqdef

    #Load a file to get some header info
    primaryheader=fits.getheader(os.path.join(filterdir,files[0]))
    filterwheel=primaryheader['FILTER']                           
    pupilwheel=primaryheader['PUPIL']                           
    
    model.meta.subarray.name = 'FULL'
    model.meta.subarray.xstart = 1
    model.meta.subarray.xsize = xd
    model.meta.subarray.ystart = 1
    model.meta.subarray.ysize = yd
    model.meta.instrument.name = instrument.upper()
    model.meta.instrument.detector = detector
    if grism=='GR700XD':
        model.meta.instrument.filter='CLEAR'
        model.meta.instrument.pupil=grism
    else:
        model.meta.instrument.filter=grism
        model.meta.instrument.pupil=pupilwheel   

    # Get the fast and slow axis directions from one of the input files
    fastaxis, slowaxis = bad_pixel_mask.badpix_from_flats.get_fastaxis(os.path.join(filterdir,files[0]))
    model.meta.subarray.fastaxis = fastaxis
    model.meta.subarray.slowaxis = slowaxis

    if 'transmission' in description:
        model.meta.offsetx=pomoutlineoffsetx
        model.meta.offsety=pomoutlineoffsety
        
    model.meta.author = author
    model.meta.description = description
    model.meta.pedigree = pedigree
    model.meta.useafter = useafter
    

    # Add HISTORY information
    package_note = ('This file was created using https://github.com/chriswillott/jwst/blob/master/makenirissgrismflats.py')    
    entry = util.create_history_entry(package_note)
    model.history.append(entry)
    package_note = ('FPA Temperature={}K'.format(fpatemp))    
    entry = util.create_history_entry(package_note)
    model.history.append(entry)
    package_note = ('Number of pixels to grow POM features={}'.format(ngrow))    
    entry = util.create_history_entry(package_note)
    model.history.append(entry)


    # Add the list of input files used to create the map
    model.history.append('DATA USED:')
    for file in files:
        totlen = len(file)
        div = np.arange(0, totlen, 60)
        for val in div:
            if totlen > (val+60):
                model.history.append(util.create_history_entry(file[val:val+60]))
            else:
                model.history.append(util.create_history_entry(file[val:]))

    # Add the do not use lists, pixel flag mappings, and user-provided
    # history text
    for history_entry in history_text:
        if history_entry != '':
            model.history.append(util.create_history_entry(history_entry))

    model.save(outfile, overwrite=True)
    print('Final reference file save to: {}'.format(outfile))

    #Add the offsetx and offsety values manually since no schema yet for transmission files
    if 'transmission' in description:
        fits.setval(outfile, 'OFFSETX', value=pomoutlineoffsetx, ext=0, after='SLOWAXIS', comment='Transmision map-detector offset in x pixels')
        fits.setval(outfile, 'OFFSETY', value=pomoutlineoffsety, ext=0, after='OFFSETX', comment='Transmision map-detector offset in y pixels')

###Main program###

#Information for file header    
author='Chris Willott'
pedigree= 'GROUND  '
useafter= '2015-11-01T00:00:00'

#Load bad pixel file
hdulist=fits.open(maskfile)
maskdata=hdulist['DQ'].data
donotuseindices=np.where(np.bitwise_and(maskdata, dqflags.pixel['DO_NOT_USE']))
numbadpix=donotuseindices[0].size

#POM feature detection will use CV3 F115W imaging and grism flat data
#Read in headers and active pixels from flat field data 
detectflatfile=os.listdir(os.path.join(indirimage,'F115W'))[0]
hdulistdetect=fits.open(detectflatfile)
detectheader=hdulistdetect['PRIMARY'].header
detectactivedata=hdulistdetect['SCI'].data[4:2044,4:2044]
activeshape=hdulistdetect['SCI'].data.shape

#Load F115W grism data for finding POM features
detectgrismdir=os.path.join(indirgrism,'F115W')
detectgrismdirlist=os.listdir(detectgrismdir)
numgrismfiles=len(detectgrismdirlist)
for l in range(numgrismfiles):
    grismfilename=os.path.join(detectgrismdir,detectgrismdirlist[l])
    hdulist=fits.open(grismfilename)
    hdulistpriheader=hdulist['PRIMARY'].header
    filterwheel=hdulistpriheader['FILTER']
    if filterwheel=='GR150C':
        hdulistgr150c=hdulist
        gr150cflatfile=grismfilename
    elif filterwheel=='GR150R':
        hdulistgr150r=hdulist
        gr150rflatfile=grismfilename
    print (grismfilename,filterwheel)    
gr150cpriheader=hdulistgr150c['PRIMARY'].header
gr150csciheader=hdulistgr150c['SCI'].header
gr150cdetectactivedata=hdulistgr150c['SCI'].data[4:2044,4:2044]
gr150rdetectactivedata=hdulistgr150r['SCI'].data[4:2044,4:2044]

#Load F150W grism data for patching POM features
patchgrismdir=os.path.join(indirgrism,'F150W')
patchgrismdirlist=os.listdir(patchgrismdir)
numgrismfiles=len(patchgrismdirlist)
for l in range(numgrismfiles):
    grismfilename=os.path.join(patchgrismdir,patchgrismdirlist[l])
    hdulist=fits.open(grismfilename)
    hdulistpriheader=hdulist['PRIMARY'].header
    filterwheel=hdulistpriheader['FILTER']
    if filterwheel=='GR150C':
        hdulistgr150c=hdulist
    elif filterwheel=='GR150R':
        hdulistgr150r=hdulist
    print (grismfilename,filterwheel)    
gr150cpriheader=hdulistgr150c['PRIMARY'].header
gr150csciheader=hdulistgr150c['SCI'].header
gr150cpatchdata=hdulistgr150c['SCI'].data
gr150rpatchdata=hdulistgr150r['SCI'].data

#Loop over filters
filterdirlist=natsort.natsorted(os.listdir(indirimage))
filterdirlist[:] = (value for value in filterdirlist if value.startswith('F'))
numfilters=len(filterdirlist)
print (numfilters,' filters for flat-field reference files')

for l in range(numfilters):

    #F090W only done at warm plateau in CV3
    if filterdirlist[l] =='F090W':
        fpatemp=43.699
    else:
        fpatemp=37.749

    filterdir=os.path.join(indirimage,filterdirlist[l])
    dirlist=natsort.natsorted(os.listdir(filterdir))
    dirlist[:] = (value for value in dirlist if value.endswith('.fits'))
    numfiles=len(dirlist)
    print(' ')
    print (l, 'Filter=',filterdirlist[l])

    #Read in file
    measureflatfile=os.path.join(filterdir,dirlist[0])
    instrument, detector = bad_pixel_mask.instrument_info(measureflatfile)
    print ('Processing file',dirlist[0])
    hdulistmeasure=fits.open(measureflatfile)
    measurepriheader=hdulistmeasure[0].header
    measuresciheader=hdulistmeasure['SCI'].header
    measuredata=hdulistmeasure['SCI'].data
    measureactivedata=measuredata[4:2044,4:2044]
    measureerr=hdulistmeasure['ERR'].data

    #Get values to normalise flats by median for detection and measurement
    normalizedetect=np.median(detectactivedata)
    normalizemeasure=np.median(measureactivedata)
    normalizegr150c=np.median(gr150cdetectactivedata)
    normalizegr150r=np.median(gr150rdetectactivedata)

    print ('Starting POM transmission generation')

    #According to Kevin Volk's CV3 report JWST-STScI-004825 all filters have <0.5 pixel shift w.r.t. F115W except F200W.
    #For F200W need to shift POM map by 1,2 pixels.
    if filterdirlist[l]=='F200W':
        xoff=-2
        yoff=1
    else:    
        xoff=0
        yoff=0

    corocen=np.array([[1061+xoff,1884+yoff],[750+xoff,1860.45+yoff],[439.45+xoff,1837.45+yoff],[129+xoff,1814.45+yoff]])
    corocenfloor=np.floor(corocen).astype('int')
    #Radii covering complete spot region to be replaced
    #For long-wave filters increase boxes by 20% because of larger PSF
    corofullradius=np.array([19,15,8,6])
    if ((filterdirlist[l] =='F277W')or(filterdirlist[l] =='F356W')or(filterdirlist[l] =='F380M')or(filterdirlist[l] =='F430M')or(filterdirlist[l] =='F444W')or(filterdirlist[l] =='F480M')):
        corofullradius=np.floor(corofullradius*1.2).astype('int')
    #Get reduced radii from those that define the complete spot region to just cover dark central region for POM transmission map
    corosizereduce=np.array([0.65,0.6,0.50,0.48])
    corodarkradius=(np.round(corofullradius*corosizereduce)).astype(int)

    #Subtract detect off grism flats, so POM features in detect appear positive 
    gr150cminusdetect=gr150cdetectactivedata/normalizegr150c-detectactivedata/normalizedetect
    gr150rminusdetect=gr150rdetectactivedata/normalizegr150r-detectactivedata/normalizedetect

    #Subtract measure off grism flats, so POM features in measure appear positive 
    gr150cminusmeasure=gr150cdetectactivedata/normalizegr150c-measureactivedata/normalizemeasure
    gr150rminusmeasure=gr150rdetectactivedata/normalizegr150r-measureactivedata/normalizemeasure

    #Initialize full size output arrays
    pommap=np.zeros(activeshape)
    pomintens=np.zeros(activeshape)

    #Iterate over the 'C' and 'R' grisms
    grismminusdetectdata=[gr150cminusdetect,gr150rminusdetect]
    grismminusmeasuredata=[gr150cminusmeasure,gr150rminusmeasure]
    for k in range(2):

        #Subtract a background from grismminusdetectdata
        #Don't need a mask excluding low and high pixels as sigma clipping will exclude bright and dark areas
        sigma_clip_forbkg = SigmaClip(sigma=3., maxiters=10)
        bkg_estimator = MedianBackground()
        bkg = Background2D(grismminusdetectdata[k], (24, 24),filter_size=(5, 5), sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator)
        grismminusdetectdata[k]=grismminusdetectdata[k]-bkg.background

        #Replace all highly negative pixels (these are dispersed POM features) with zeros
        v=np.where(grismminusdetectdata[k]<-0.02)
        grismminusdetectdata[k][v]=0.0

        #Run source extraction twice
        #Do not use a filter as would make single bright pixels appear as sources
        #Do first run to find small things 
        threshold_small = detect_threshold(grismminusdetectdata[k], nsigma=3.0)
        segm_small = detect_sources(grismminusdetectdata[k], threshold_small, npixels=4, connectivity=4, filter_kernel=None) 
        #Do second run to find large things with lower significance and merge the two
        threshold_large = detect_threshold(grismminusdetectdata[k], nsigma=1.5)
        segm_large = detect_sources(grismminusdetectdata[k], threshold_large, npixels=100, connectivity=8, filter_kernel=None) 

        #Remove some sources from large segmentation image
        catsegm_large = source_properties(grismminusdetectdata[k], segm_large)
        numsource=np.array(catsegm_large.label).size
        for ct in range(numsource):
            #Remove sources in lower 450 pixel strip because POM not in focus so can't be real. 
            if (catsegm_large.ycentroid[ct].value<450.0):
                segm_large.remove_labels(labels=catsegm_large.label[ct])
            #Remove sources in upper 100 pixel strip because due to fringing
            elif (catsegm_large.ycentroid[ct].value>1940):
                segm_large.remove_labels(labels=catsegm_large.label[ct])
            #Remove bad detector region at x=12, y=1536
            elif ((catsegm_large.ycentroid[ct].value>1520)&(catsegm_large.ycentroid[ct].value<1555)&(catsegm_large.xcentroid[ct].value<25)):
                segm_large.remove_labels(labels=catsegm_large.label[ct])

        #Remove some sources from small segmentation image
        catsegm_small = source_properties(grismminusdetectdata[k], segm_small)
        numsource=np.array(catsegm_small.label).size
        for ct in range(numsource):

            #Remove all sources in lower 450 pixel strip because POM not in focus so can't be real. 
            if (catsegm_small.ycentroid[ct].value<450.0):
                segm_small.remove_labels(labels=catsegm_small.label[ct])
            #Remove small sources (area<6 pixels) except in top part of detector because POM not in focus so can't be real. 
            elif ((catsegm_small.ycentroid[ct].value>450.0 and catsegm_small.ycentroid[ct].value<1700.0 and catsegm_small.area[ct].value<6)):
                segm_small.remove_labels(labels=catsegm_small.label[ct])
            #Remove bad detector region at x=12, y=1536
            elif ((catsegm_small.ycentroid[ct].value>1520)&(catsegm_small.ycentroid[ct].value<1555)&(catsegm_small.xcentroid[ct].value<25)):
                segm_small.remove_labels(labels=catsegm_small.label[ct])

        #some intermediate outputs for testing        
        #fits.writeto('testgrismminusdetectdata{}.fits'.format(k), grismminusdetectdata[k], overwrite=True)
        #fits.writeto('testsegsmall{}.fits'.format(k), segm_small.data, overwrite=True)
        #fits.writeto('testseglarge{}.fits'.format(k), segm_large.data, overwrite=True)
            
        #Subtract a background from grismminusmeasuredata
        #Don't need a mask excluding low and high pixels as sigma clipping will exclude bright and dark areas
        sigma_clip_forbkg = SigmaClip(sigma=3., maxiters=10)
        bkg_estimator = MedianBackground()
        bkg = Background2D(grismminusmeasuredata[k], (24, 24),filter_size=(5, 5), sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator)
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
        pommap2040=ndimage.convolve(pommap2040, kern, mode='constant', cval=0.0)
        pommap2040[np.where(pommap2040>1.0)]=1.0

        #Merge the GR150C and GR150R images and shift if necessary when placing within the full detector
        if k==0:
            pommap[4+yoff:2044+yoff,4+xoff:2044+xoff]=pommap2040
            pomintens[4:2044,4:2044]=grismminusmeasuredata[k]*pommap[4:2044,4:2044]
        else:
            pommap[4+yoff:2044+yoff,4+xoff:2044+xoff]=np.maximum(pommap[4+yoff:2044+yoff,4+xoff:2044+xoff],pommap2040)        
            pomintens[4:2044,4:2044]=np.maximum(pomintens[4:2044,4:2044],(grismminusmeasuredata[k]*pommap[4:2044,4:2044]))

    #Set constant high value in central regions of coronagraphic spots as too noisy to measure in flats 
    #Iterate over the four spots
    for k in range(4):
        coroaperture = CircularAperture(corocen[k,:], r=corodarkradius[k])
        coro_obj_mask = coroaperture.to_mask(method='center')
        coro_obj_mask_corr=2.0*coro_obj_mask.data
        pomintens[corocenfloor[k,1]-corodarkradius[k]:corocenfloor[k,1]+corodarkradius[k]+1,corocenfloor[k,0]-corodarkradius[k]:corocenfloor[k,0]+corodarkradius[k]+1]+=coro_obj_mask_corr

    #Replace pixels with intensity>0.98 to 1.00 as even though centres of spots in CV3 had values of a few percent this was probably scattered
    w=np.where(pomintens>0.98)
    pomintens[w]=1.00

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
    print ('Total fraction of sensitivity decrease in accounted POM features=',fractionfluxlost)

    #Will output intensity files as transmission so do 1-intens
    pomtransmission=1.0-pomintensfixbadpix

    #Load F200W POM outline file derived in flatsplot.py
    hdulistpomoutline=fits.open(pomoutlinefile)
    pomoutlineheader=hdulistpomoutline[0].header
    fullpomtransmission=hdulistpomoutline[1].data
    pomoutlineoffsetx=pomoutlineheader['OFFSETX']
    pomoutlineoffsety=pomoutlineheader['OFFSETY']
    
    # POM outline file is for F200W so apply filter-dependent shifts for all others
    if filterdirlist[l]!='F200W':
        xofffilter=2
        yofffilter=-1
        fullpomtransmission[0:yofffilter,xofffilter:]=fullpomtransmission[-1*yofffilter:,0:-1*xofffilter]
        
    fullpomtransmission[pomoutlineoffsety:(pomoutlineoffsety+2048),pomoutlineoffsetx:(pomoutlineoffsetx+2048)]=pomtransmission
    fullpomtransmissionerr=fullpomtransmission/100.0
    
    filterwheel=measurepriheader['FILTER']                           
    pupilwheel=measurepriheader['PUPIL']                           

    #Add information about inputs to output POM transmission file header
    #hdup=fits.PrimaryHDU(data=None)
    #    hdusci=fits.ImageHDU(data=fullpomtransmission,header=gr150csciheader,name='SCI')
    #    hdup.header.append(('DATE',(datetime.isoformat(datetime.utcnow())),'Date this file was created (UTC)'),end=True)
    #    hdup.header.append(('TELESCOP','JWST','Telescope used to acquire the data'),end=True)
    #    hdup.header.append(('PEDIGREE','GROUND','The pedigree of the reference file'),end=True)
    #    hdup.header.append(('DESCRIP','This is a POM transmission reference file.','Description of the reference file'),end=True)
    #    hdup.header.append(('AUTHOR','Chris Willott','Author of the reference file'),end=True)
    #    hdup.header.append(('', ''),end=True)
    #    hdup.header.append(('', 'Instrument configuration information'),end=True)
    #    hdup.header.append(('', ''),end=True)
    #    hdup.header.append(('INSTRUME', 'NIRISS', 'Instrument used to acquire the data'),end=True)
    #    hdup.header.append(('DETECTOR', 'NIS', 'Name of detector used to acquire the data'),end=True)
    #    hdup.header.append(('FILTER', filterwheel, 'Name of filter element used'),end=True)
    #    hdup.header.append(('PUPIL', pupilwheel, 'Name of the pupil element used'),end=True)
    #    hdup.header.append(('SUBARRAY', 'FULL', 'Subarray used'),end=True)
    #    hdup.header.append(('', ''),end=True)
    #    hdup.header.append(('', 'POM Fitting Information'),end=True)
    #    hdup.header.append(('', ''),end=True)
    #    hdup.header.append(('DETPOM',os.path.basename(detectflatfile),'Direct flat for POM detection'),end=True)
    #    hdup.header.append(('MEASPOM',os.path.basename(measureflatfile),'Direct flat for POM measurement'),end=True)
    #    hdup.header.append(('CFLATPOM',os.path.basename(gr150cflatfile),'GR150C flat for POM'),end=True)
    #    hdup.header.append(('RFLATPOM',os.path.basename(gr150rflatfile),'GR150R flat for POM'),end=True)
    #    hdup.header.append(('BPMPOM',os.path.basename(maskfile),'Bad pixel mask for POM'),end=True)
    #    hdup.header.append(('NGROWPOM',ngrow,'Number of pixels to grow POM features'),end=True)
    #    hdup.header.append(('OFFSETX',pomoutlineoffsetx,'transmision map-detector offset in x pixels'),end=True)
    #    hdup.header.append(('OFFSETY',pomoutlineoffsety,'transmision map-detector offset in y pixels'),end=True)
    #    hdulistout=fits.HDUList([hdup,hdusci])

    #Define DQ_DEF binary table HDU
    flagtable=np.rec.array([
           ( 0,      0, 'GOOD',            'Good pixel')],
           formats='int8,int8,a40,a80',
           names='Bit,Value,Name,Description')
    dqdef = flagtable

    #Set up DQ array
    dq=np.zeros(fullpomtransmission.shape,dtype=np.int8)

    history = []
    hdu = fits.PrimaryHDU()
    all_files=[os.path.basename(measureflatfile),os.path.basename(detectflatfile),os.path.basename(gr150cflatfile),os.path.basename(gr150rflatfile),os.path.basename(maskfile)]

    description='This is a pick-off mirror transmission reference file.'

    #Output 2 GR150 grism files per filter and a GR700XD grism file for F150W in reference file format
    grisms=['GR150C','GR150R','GR700XD']
    for k in range(3):
        if ((k<2) or (filterdirlist[l]=='F200W')):
            outfile = 'jwst_niriss_cv3_pomtransmission_{}_{}.fits'.format(grisms[k],filterdirlist[l])
            outdirwithfilter=os.path.join(outdir, filterdirlist[l])
            output_file = os.path.join(outdirwithfilter,outfile)
            if not os.path.exists(outdirwithfilter):
                os.makedirs(outdirwithfilter)
            hdu_list = fits.HDUList([hdu])
            save_final_map(fullpomtransmission, dq, fullpomtransmissionerr, dqdef, instrument.upper(), detector.upper(), hdu_list, filterdir, grisms[k],
                        all_files, author, description, pedigree, useafter, fpatemp, history, pomoutlineoffsetx, pomoutlineoffsety, ngrow, output_file)
    
    #Output POM transmission file
    #print ('Writing',output_file)
    #hdulistout.writeto(output_file,overwrite=True)

    #End POM transmission file making
    
    ##################################################### 
    #Start grism flat field reference file making
    
    print ('Starting grism flat field generation')
    #Do correction for all POM features using POM transmission image
    measuredata/=pomtransmission
    measureerr/=pomtransmission

    #Coronagraphic spots have too low S/N in POM transmission image so  will replace with regions from locally normalized grism flats.
    #Iterate over the four spots
    for k in range(4):
        #local normalization in nearby annulus just outside coronagraphic spots
        coro_annulus_aperture = CircularAnnulus(corocen[k,:], r_in=1.2*corofullradius[k], r_out=2.0*corofullradius[k])
        coro_annulus_masks = coro_annulus_aperture.to_mask(method='center')

        coro_annulus_measuredata = coro_annulus_masks.multiply(measuredata)
        coro_annulus_mask = coro_annulus_masks.data
        coro_annulus_measuredata_1d = coro_annulus_measuredata[coro_annulus_mask > 0]
        coro_annulus_measuredata_clipped=sigma_clip(coro_annulus_measuredata_1d,sigma=3.0,maxiters=3)
        coro_annulus_measuredata_median=np.ma.median(coro_annulus_measuredata_clipped)

        coro_annulus_gr150cpatchdata = coro_annulus_masks.multiply(gr150cpatchdata)
        coro_annulus_gr150cpatchdata_1d = coro_annulus_gr150cpatchdata[coro_annulus_mask > 0]
        coro_annulus_gr150cpatchdata_clipped=sigma_clip(coro_annulus_gr150cpatchdata_1d,sigma=3.0,maxiters=3)
        coro_annulus_gr150cpatchdata_median=np.ma.median(coro_annulus_gr150cpatchdata_clipped)

        coro_annulus_gr150rpatchdata = coro_annulus_masks.multiply(gr150rpatchdata)
        coro_annulus_gr150rpatchdata_1d = coro_annulus_gr150rpatchdata[coro_annulus_mask > 0]
        coro_annulus_gr150rpatchdata_clipped=sigma_clip(coro_annulus_gr150rpatchdata_1d,sigma=3.0,maxiters=3)
        coro_annulus_gr150rpatchdata_median=np.ma.median(coro_annulus_gr150rpatchdata_clipped)

        coro_scale_gr150c=coro_annulus_measuredata_median/coro_annulus_gr150cpatchdata_median
        coro_scale_gr150r=coro_annulus_measuredata_median/coro_annulus_gr150rpatchdata_median

        coro_scaled_gr150cpatchdata=gr150cpatchdata*coro_scale_gr150c
        coro_scaled_gr150rpatchdata=gr150rpatchdata*coro_scale_gr150r

        xlo=corocenfloor[k,0]-corofullradius[k]
        xhi=corocenfloor[k,0]+corofullradius[k]+1
        ylo=corocenfloor[k,1]-corofullradius[k]
        yhi=corocenfloor[k,1]+corofullradius[k]+1

        #Replace with maximum value from either C or R grism
        measuredata[ylo:yhi,xlo:xhi]=np.maximum(coro_scaled_gr150cpatchdata[ylo:yhi,xlo:xhi],coro_scaled_gr150rpatchdata[ylo:yhi,xlo:xhi])
        #Set S/N=100 for all replaced regions of spots because of possible systematics
        measureerr[ylo:yhi,xlo:xhi]=measuredata[ylo:yhi,xlo:xhi]/100.0

    #Set DQ flags for bad pixels    
    #UNRELIABLE_FLAT for all pixels at active pixel border because 10% higher in flats
    unrelflat=np.zeros(measuredata.shape,dtype=int)
    unrelflat[4:5,4:2044]=1
    unrelflat[2043:2044,4:2044]=1
    unrelflat[5:2043,4:5]=1
    unrelflat[5:2043,2043:2044]=1

    #UNRELIABLE_FLAT for all pixels with values <0.1
    unrelflat[np.where(measuredata<0.1)]=1

    #Reference pixel flag for all reference pixels
    refpix=np.zeros(measuredata.shape,dtype=int)
    refpix[:4,:]=1  
    refpix[2044:,:]=1
    refpix[:,:4]=1   
    refpix[:,2044:]=1

    #Set all reference pixels to one
    measuredata[np.where(refpix==1)]=1.0
    measureerr[np.where(refpix==1)]=1.0

    #Set all negative pixels to zero
    measureerr[np.where(measuredata<0.0)]=0.0
    measuredata[np.where(measuredata<0.0)]=0.0

    #Define DQ_DEF binary table HDU
    flagtable=np.rec.array([
           ( 0,      0, 'GOOD',            'Good pixel'),
           ( 0,      1, 'DO_NOT_USE',      'Bad pixel. Do not use for science or calibration'),
           ( 1,      2, 'NO_FLAT_FIELD',   'Flat field cannot be measured'),
           ( 2,      4, 'UNRELIABLE_FLAT', 'Flat variance large'),
           ( 4,      8, 'REFERENCE_PIXEL', 'All reference pixels'   )],
           formats='int8,int8,a40,a80',
           names='Bit,Value,Name,Description')
    dqdef = flagtable

    #Set up DQ array
    dq=np.zeros(measuredata.shape,dtype=np.int8)

    #Set UNRELIABLE_FLAT and REFERENCE_PIXEL flagged pixels to DO_NOT_USE 
    bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'DO_NOT_USE')][0]
    flagarray=np.ones(measuredata.shape, dtype=np.int8)*bitvalue
    w=np.where((unrelflat>0)|(refpix>0))
    dq[w]=np.bitwise_or(dq[w],flagarray[w])
    #dq[w]=flagarray[w]

    bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_FLAT')][0]
    flagarray=np.ones(measuredata.shape, dtype=np.int8)*bitvalue
    w=np.where(unrelflat>0)
    dq[w]=np.bitwise_or(dq[w],flagarray[w])
    #dq[w]=flagarray[w]

    bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'REFERENCE_PIXEL')][0]
    flagarray=np.ones(measuredata.shape, dtype=np.int8)*bitvalue
    w=np.where(refpix>0)
    dq[w]=np.bitwise_or(dq[w],flagarray[w])
    #dq[w]=flagarray[w]

    history = []
    hdu = fits.PrimaryHDU()
    all_files=[os.path.basename(measureflatfile),os.path.basename(detectflatfile),os.path.basename(detectgrismdirlist[0]),os.path.basename(detectgrismdirlist[1]),os.path.basename(patchgrismdirlist[0]),os.path.basename(patchgrismdirlist[1]),os.path.basename(maskfile)]

    description='This is a pixel flat reference file.'

    #Output 2 GR150 grism files per filter and a GR700XD grism file for F200W in reference file format
    grisms=['GR150C','GR150R','GR700XD']
    for k in range(3):
        if ((k<2) or (filterdirlist[l]=='F200W')):
            outfile = 'jwst_niriss_cv3_grismflat_{}_{}.fits'.format(grisms[k],filterdirlist[l])
            outdirwithfilter=os.path.join(outdir, filterdirlist[l])
            output_file = os.path.join(outdirwithfilter,outfile)
            if not os.path.exists(outdirwithfilter):
                os.makedirs(outdirwithfilter)
            hdu_list = fits.HDUList([hdu])
            save_final_map(measuredata, dq, measureerr, dqdef, instrument.upper(), detector.upper(), hdu_list, filterdir, grisms[k],
                               all_files, author, description, pedigree, useafter, fpatemp, history, pomoutlineoffsetx, pomoutlineoffsety, ngrow, output_file)

