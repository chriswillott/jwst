#!/usr/bin/env python

#Generate NIRISS imaging flat field reference files.
#Uses jwst and jwst_reffiles python packages
#Normalize, sigma clip and combine multiple flat integrations/exposures.
#The flat field will be normalized to a sigma-clipped average value of one as per the CALWEBB_IMAGE2 definition.
#No surface fitting to account for uneven illumination is applied since the CV3 OSIM illumination pattern is unknown.
#Four unusual blob regions in some filter images are replaced with the F200W flat that doesn't show them. 
#Inputs: place all flat field slope_rateints.fits and/or slope_rate.fits files in one sub-directory per filter under the input directory indir, e.g. ./slope/F150W/
#Outputs: final flat field images will be output to one sub-directory per filter under the output directory, e.g. ./imageflatreffiles/F150W/

#Usage, e.g.
#makenirissimagingflats.py --indir='./slope' --outdir='./imageflatreffiles'

import numpy as np
import optparse, sys
import os
import astropy.io.fits as fits
from astropy.stats import SigmaClip,sigma_clip,sigma_clipped_stats
from copy import deepcopy
from photutils import detect_threshold,detect_sources,Background2D, MedianBackground,source_properties, CircularAperture, CircularAnnulus
import natsort
from jwst.datamodels import FlatModel, util
from jwst_reffiles.bad_pixel_mask import bad_pixel_mask

# Command line options
op = optparse.OptionParser()
op.add_option("--indir")
op.add_option("--outdir")

o, a = op.parse_args()
if a:
    print (sys.stderr, "unrecognized option: ",a)
    sys.exit(-1)

indir=o.indir
outdir=o.outdir
if indir==None:
    indir='./'
if outdir==None:
    outdir='./'

if not os.path.exists(outdir):
    os.makedirs(outdir)

def save_final_map(flat_map, flat_dq, flat_err, dqdef, instrument, detector, hdulist, filterdir, files,
                       author, description, pedigree,useafter, fpatemp, history_text, outfile):
    """Save a flat field map into a CRDS-formatted reference file
    Parameters
    ----------
    flat_map : numpy.ndarray
        2D flat-field array
    flat_dq : numpy.ndarray
        2D flat-field DQ array
    flat_err : numpy.ndarray
        2D flat-field error array
    dqdef : numpy.ndarray
        binary table of DQ definitions
    instrument : str
        Name of instrument associated with the flat-field array
    detector : str
        Name of detector associated with the flat-field array
    hdulist : astropy.fits.HDUList
        HDUList containing "extra" fits keywords
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
    yd, xd = flat_map.shape

    # Initialize the MaskModel using the hdu_list, so the new keywords will
    # be populated
    model = FlatModel(hdulist)
    model.data = flat_map
    model.dq = flat_dq
    model.err = flat_err
    model.dq_def = dqdef

    #Load a file to get some header info
    primaryheader=fits.getheader(os.path.join(filterdir,files[0]))
    filterwheel=primaryheader['FILTER']                           
    pupilwheel=primaryheader['PUPIL']                           
    
    model.meta.reftype = 'FLAT'
    model.meta.subarray.name = 'FULL'
    model.meta.subarray.xstart = 1
    model.meta.subarray.xsize = xd
    model.meta.subarray.ystart = 1
    model.meta.subarray.ysize = yd
    model.meta.instrument.name = instrument.upper()
    model.meta.instrument.detector = detector
    model.meta.instrument.filter=filterwheel
    model.meta.instrument.pupil=pupilwheel

    # Get the fast and slow axis directions from one of the input files
    fastaxis, slowaxis = bad_pixel_mask.badpix_from_flats.get_fastaxis(os.path.join(filterdir,files[0]))
    model.meta.subarray.fastaxis = fastaxis
    model.meta.subarray.slowaxis = slowaxis

    model.meta.author = author
    model.meta.description = description
    model.meta.pedigree = pedigree
    model.meta.useafter = useafter    

    # Add HISTORY information
    package_note = ('This file was created using https://github.com/chriswillott/jwst/blob/master/makenirissimagingflats.py')    
    entry = util.create_history_entry(package_note)
    model.history.append(entry)
    package_note = ('FPA Temperature={}K'.format(fpatemp))    
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
    print('Final flat reference file save to: {}'.format(outfile))
    

###Main program###

#Information for file header    
author='Chris Willott'
description='This is a pixel flat reference file.'
pedigree= 'GROUND  '
useafter= '2015-11-01T00:00:00'

#Get list of directories with filter names
filterdirlist=natsort.natsorted(os.listdir(indir))
filterdirlist[:] =(value for value in filterdirlist if value.startswith('F'))
filterdirlist=np.array(filterdirlist)

#Reorder to put F200W first since that will be used for patching some other filters
w=np.where(filterdirlist=='F200W')
filterdirlist[w]=filterdirlist[0]
filterdirlist[0]='F200W'
numfilters=len(filterdirlist)
print (numfilters,' filters for flat-field reference files')

#Iterate over filters
for l in range(numfilters):

    #F090W only done at warm plateau in CV3
    if filterdirlist[l] =='F090W':
        fpatemp=43.699
    else:
        fpatemp=37.749
        
    normdata3d=[]
    #Get list of files in this directory
    filterdir=os.path.join(indir,filterdirlist[l])
    dirlist=natsort.natsorted(os.listdir(filterdir))
    dirlist[:] = (value for value in dirlist if value.endswith('.fits'))
    numfiles=len(dirlist)
    print(' ')
    print ('Filter=',filterdirlist[l],' with ',numfiles,' files')

    if numfiles>0:
        for j in range(numfiles):
            #Read in file
            slopefile=os.path.join(filterdir,dirlist[j])
            instrument, detector = bad_pixel_mask.instrument_info(slopefile)
            print ('Processing file ',dirlist[j])
            hdulist=fits.open(slopefile)
            header=hdulist[0].header
            data=hdulist['SCI'].data
            err=hdulist['ERR'].data
            #Stack data and error arrays into 3D arrays
            if '_rateints.fits' in slopefile:
                nint=data.shape[0]
                for k in range(nint):
                    #Don't include reference pixels or pixels bordering reference pixels in statistics for normalization
                    datafornormalize=data[k,5:2043,5:2043]
                    normdata=data[k,:,:]/np.median(datafornormalize)
                    normerr=err[k,:,:]/np.median(datafornormalize)
                    if len(normdata3d)==0:
                        normdata3d=normdata
                        normerr3d=normerr
                    else:
                        normdata3d=np.dstack((normdata3d,normdata))
                        normerr3d=np.dstack((normerr3d,normerr))
            elif '_rate.fits' in slopefile:
                #Don't include reference pixels or pixels bordering reference pixels in statistics for normalization
                datafornormalize=data[5:2043,5:2043]
                normdata=data/np.median(datafornormalize)
                normerr=err/np.median(datafornormalize)
                if len(normdata3d)==0:
                    normdata3d=normdata
                    normerr3d=normerr
                else:
                    normdata3d=np.dstack((normdata3d,normdata))
                    normerr3d=np.dstack((normerr3d,normerr))

        #Combine stacks clipping outliers and masked arrays
        #If 10 or more total integrations use sigma clipping with sigma=3.0 where sigma is std dev from all integrations
        #If less than 10 total integrations exclude everything >5sigma from median using sigma from error array
        numtotint=normdata3d.shape[2]
        print ('Total of ',numtotint,' integrations for this filter')       
        if numtotint<10:
            numsigma=5
            meddata3d=np.median(normdata3d,axis=2,keepdims=True)
            datadiffs=normdata3d-meddata3d
            datadiffsdiverr=datadiffs/normerr3d
            datadiffsdiverrmasked=np.ma.masked_greater_equal(datadiffsdiverr,numsigma)
            clippeddata3d=np.ma.masked_array(normdata3d,datadiffsdiverrmasked.mask)
            clippederr3d=np.ma.masked_array(normerr3d,clippeddata3d.mask)
        else:
            numsigma=3.0
            clippeddata3d=sigma_clip(normdata3d,sigma=numsigma,maxiters=2,axis=2,cenfunc='median')
            clippederr3d=np.ma.masked_array(normerr3d,clippeddata3d.mask)
        #For data array use mean of clipped array
        meanclippeddata=np.ma.mean(clippeddata3d,axis=2)
        #For error array add errors in quadrature and divide by number of unmasked samples
        meanclippederr=(1.0/(np.sum((~clippederr3d.mask),axis=2)))*(np.ma.sum((clippederr3d**2),axis=2))**0.5

        #Patch unusual blob regions in some filter images, replacing with the F200W flat that doesn't show them.
        if filterdirlist[l] != 'F200W':
            hdulistf200w=fits.open('jwst_niriss_cv3_imageflat_F200W.fits')
            dataf200w=hdulistf200w['SCI'].data
            #These are x,y positions for photutils
            blobcen=np.array([[1218,1590],[1888,1044],[1975,963],[179,1865]])
            #Radii covering complete blob regions to be replaced   
            blobradius=np.array([13,10,13,6])
            for k in range(4):
                if ((k==0)or((k==2)and((filterdirlist[l]=='F090W')or(filterdirlist[l]=='F140M')or(filterdirlist[l]=='F150W')or(filterdirlist[l]=='F158M')or(filterdirlist[l]=='F277W')))or(((k==1)or(k==3))and((filterdirlist[l]=='F090W')or(filterdirlist[l]=='F115W')or(filterdirlist[l]=='F140M')or(filterdirlist[l]=='F150W')or(filterdirlist[l]=='F158M')or(filterdirlist[l]=='F277W')))):
                    #local normalization in nearby annulus just outside blobs
                    blob_annulus_aperture = CircularAnnulus(blobcen[k,:], r_in=1.2*blobradius[k], r_out=2.0*blobradius[k])
                    blob_annulus_masks = blob_annulus_aperture.to_mask(method='center')

                    blob_annulus_meanclippeddata = blob_annulus_masks.multiply(meanclippeddata)
                    blob_annulus_mask = blob_annulus_masks.data
                    blob_annulus_meanclippeddata_1d = blob_annulus_meanclippeddata[blob_annulus_mask > 0]
                    blob_annulus_meanclippeddata_clipped=sigma_clip(blob_annulus_meanclippeddata_1d,sigma=3.0,maxiters=3)
                    blob_annulus_meanclippeddata_median=np.ma.median(blob_annulus_meanclippeddata_clipped)

                    blob_annulus_dataf200w = blob_annulus_masks.multiply(dataf200w)
                    blob_annulus_dataf200w_1d = blob_annulus_dataf200w[blob_annulus_mask > 0]
                    blob_annulus_dataf200w_clipped=sigma_clip(blob_annulus_dataf200w_1d,sigma=3.0,maxiters=3)
                    blob_annulus_dataf200w_median=np.ma.median(blob_annulus_dataf200w_clipped)

                    blob_scale_gr150c=blob_annulus_meanclippeddata_median/blob_annulus_dataf200w_median

                    blob_scaled_dataf200w=dataf200w*blob_scale_gr150c

                    xlo=blobcen[k,0]-blobradius[k]
                    xhi=blobcen[k,0]+blobradius[k]+1
                    ylo=blobcen[k,1]-blobradius[k]
                    yhi=blobcen[k,1]+blobradius[k]+1

                    #Replace with F200W values
                    meanclippeddata[ylo:yhi,xlo:xhi]=blob_scaled_dataf200w[ylo:yhi,xlo:xhi]
                    #Set S/N=100 for all replaced regions of blobs because of possible systematics
                    meanclippederr[ylo:yhi,xlo:xhi]=meanclippeddata[ylo:yhi,xlo:xhi]/100.0

            
        #Renormalize using clipping after excluding all pixels with values <0.1  
        fornormalize=meanclippeddata[5:2043,5:2043]
        quicknorm=fornormalize/np.median(fornormalize)
        w=np.where(quicknorm>0.10)
        fornormalize=fornormalize[w].flatten()
        fornormclipped=sigma_clip(fornormalize,sigma=3.0,maxiters=3,cenfunc='median')
        meanfornormclipped=np.ma.mean(fornormclipped)
        normdata=meanclippeddata/meanfornormclipped
        normdata=normdata.data
        normerr=meanclippederr/meanfornormclipped
        normerr=normerr.data

        #UNRELIABLE_FLAT for all pixels at active pixel border because 10% higher in flats
        unrelflat=np.zeros(normdata.shape,dtype=int)
        unrelflat[4:5,4:2044]=1
        unrelflat[2043:2044,4:2044]=1
        unrelflat[5:2043,4:5]=1
        unrelflat[5:2043,2043:2044]=1

        #UNRELIABLE_FLAT for all pixels with values <0.1
        unrelflat[np.where(normdata<0.1)]=1

        #Reference pixel flag for all reference pixels
        refpix=np.zeros(normdata.shape,dtype=int)
        refpix[:4,:]=1  
        refpix[2044:,:]=1
        refpix[:,:4]=1   
        refpix[:,2044:]=1
       
        #Set all reference pixels to one
        normdata[np.where(refpix==1)]=1.0
        normerr[np.where(refpix==1)]=1.0

        #Set all negative pixels to zero
        normerr[np.where(normdata<0.0)]=0.0
        normdata[np.where(normdata<0.0)]=0.0
        
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
        #dqdef = fits.BinTableHDU(flagtable,name='DQ_DEF  ',ver=1)

        #Set up DQ array
        dq=np.zeros(normdata.shape,dtype=np.int8)

        #Set UNRELIABLE_FLAT and REFERENCE_PIXEL flagged pixels to DO_NOT_USE 
        bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'DO_NOT_USE')][0]
        flagarray=np.ones(normdata.shape, dtype=np.int8)*bitvalue
        w=np.where((unrelflat>0)|(refpix>0))
        dq[w]=np.bitwise_or(dq[w],flagarray[w])
        #dq[w]=flagarray[w]

        bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'UNRELIABLE_FLAT')][0]
        flagarray=np.ones(normdata.shape, dtype=np.int8)*bitvalue
        w=np.where(unrelflat>0)
        dq[w]=np.bitwise_or(dq[w],flagarray[w])
        #dq[w]=flagarray[w]

        bitvalue=flagtable['Value'][np.where(flagtable['Name']==b'REFERENCE_PIXEL')][0]
        flagarray=np.ones(normdata.shape, dtype=np.int8)*bitvalue
        w=np.where(refpix>0)
        dq[w]=np.bitwise_or(dq[w],flagarray[w])
        #dq[w]=flagarray[w]
        
        history = []
        hdu = fits.PrimaryHDU()
        all_files=dirlist
        
        #Output files in reference file format
        outfile = 'jwst_niriss_cv3_imageflat_{}.fits'.format(filterdirlist[l])
        #outfile = 'jwst_niriss_cv3_imageflat_{}_{}.fits'.format(filterdirlist[l], current_time)
        outdirwithfilter=os.path.join(outdir, filterdirlist[l])
        output_file = os.path.join(outdirwithfilter,outfile)
        if not os.path.exists(outdirwithfilter):
            os.makedirs(outdirwithfilter)
        hdu_list = fits.HDUList([hdu])
        save_final_map(normdata, dq, normerr, dqdef, instrument.upper(), detector.upper(), hdu_list, filterdir, all_files, author, description, pedigree, useafter, fpatemp, history, output_file)

