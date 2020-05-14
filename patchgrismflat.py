#!/usr/bin/env python

#Generate NIRISS WFSS grism flat field by correcting the imaging flat with the POM transmission file.
#Coronagraphic spots have too low S/N in POM transmission image so will replace with regions from locally normalized grism flats.
#Use the F150W grism flats that separate out the dispersed spectra and direct images of sources for all filters.

#Usage:
#patchgrismflat.py --directflat='./slope/NIST74-F115-FL-6003020347_1_496_SE_2016-01-03T02h19m52_slope_norm.fits'   --gr150cflat='./slope/NIST74-F150-C-FL-6003051254_1_496_SE_2016-01-03T05h27m32_slope_norm.fits' --gr150rflat='./slope/NIST74-F150-R-FL-6003054311_1_496_SE_2016-01-03T05h59m12_slope_norm.fits' --pomfile='/Users/willottc/niriss/detectors/willott_reference_files/jwst_niriss_cv3_F115W_pomtransmission.fits'  --outfile='/Users/willottc/niriss/detectors/willott_reference_files/jwst_niriss_cv3_grismflat_F115W.fits'

import numpy as np
import optparse
import os, os.path
import astropy.io.fits as fits
from astropy.stats import sigma_clip,sigma_clipped_stats
from copy import deepcopy
from photutils import detect_threshold,detect_sources,Background2D, MedianBackground,source_properties, CircularAperture, CircularAnnulus

# Command line options
op = optparse.OptionParser()
op.add_option("--directflat")
op.add_option("--gr150cflat")
op.add_option("--gr150rflat")
op.add_option("--pomfile")
op.add_option("--outfile")

o, a = op.parse_args()
if a:
    print (sys.syserr, "unrecognized option: ",a)
    sys.exit(-1)

directflatfile=o.directflat
gr150cflatfile=o.gr150cflat
gr150rflatfile=o.gr150rflat
pomfile=o.pomfile
outfile=o.outfile

#Read in data
hdulist1=fits.open(directflatfile)
directpriheader=hdulist1['PRIMARY'].header
directsciheader=hdulist1['SCI'].header
directdata=hdulist1['SCI'].data
hdulist2=fits.open(gr150cflatfile)
gr150cdata=hdulist2['SCI'].data
hdulist3=fits.open(gr150rflatfile)
gr150rdata=hdulist3['SCI'].data
hdulist4=fits.open(pomfile)
pomtransmission=hdulist4['SCI'].data
invertpomtransmission=1.0-pomtransmission

#Do correction for all POM features using POM transmission image
directdata/=(1.0-invertpomtransmission)

#Coronagraphic spots have too low S/N in POM transmission image so  will replace with regions from locally normalized grism flats.
#According to Kevin Volk's CV3 report JWST-STScI-004825 all filters have <0.5 pixel shift w.r.t. F115W except F200W.
#For F200W need to shift coronagraphic spot region by 1,2 pixels.
if directpriheader['PUPIL']=='F200W':
    xoff=-2
    yoff=1
else:    
    xoff=0
    yoff=0
#F115W spot positions
#Note these are x,y because that is used by photutils
corocen=np.array([[1061+xoff,1884+yoff],[750+xoff,1860.45+yoff],[439.45+xoff,1837.45+yoff],[129+xoff,1814.45+yoff]])
corocenfloor=np.floor(corocen).astype('int')
cororadius=np.array([19,14,8,6])

#Iterate over the four spots
for k in range(4):
    #local normalization in nearby annulus just outside coronagraphic spots
    coro_annulus_aperture = CircularAnnulus(corocen[k,:], r_in=1.2*cororadius[k], r_out=2.0*cororadius[k])
    coro_annulus_masks = coro_annulus_aperture.to_mask(method='center')

    coro_annulus_directdata = coro_annulus_masks.multiply(directdata)
    coro_annulus_mask = coro_annulus_masks.data
    coro_annulus_directdata_1d = coro_annulus_directdata[coro_annulus_mask > 0]
    coro_annulus_directdata_clipped=sigma_clip(coro_annulus_directdata_1d,sigma=3.0,maxiters=3)
    coro_annulus_directdata_median=np.ma.median(coro_annulus_directdata_clipped)

    coro_annulus_gr150cdata = coro_annulus_masks.multiply(gr150cdata)
    coro_annulus_gr150cdata_1d = coro_annulus_gr150cdata[coro_annulus_mask > 0]
    coro_annulus_gr150cdata_clipped=sigma_clip(coro_annulus_gr150cdata_1d,sigma=3.0,maxiters=3)
    coro_annulus_gr150cdata_median=np.ma.median(coro_annulus_gr150cdata_clipped)

    coro_annulus_gr150rdata = coro_annulus_masks.multiply(gr150rdata)
    coro_annulus_gr150rdata_1d = coro_annulus_gr150rdata[coro_annulus_mask > 0]
    coro_annulus_gr150rdata_clipped=sigma_clip(coro_annulus_gr150rdata_1d,sigma=3.0,maxiters=3)
    coro_annulus_gr150rdata_median=np.ma.median(coro_annulus_gr150rdata_clipped)

    coro_scale_gr150c=coro_annulus_directdata_median/coro_annulus_gr150cdata_median
    coro_scale_gr150r=coro_annulus_directdata_median/coro_annulus_gr150rdata_median

    coro_scaled_gr150cdata=gr150cdata*coro_scale_gr150c
    coro_scaled_gr150rdata=gr150rdata*coro_scale_gr150r

    xlo=corocenfloor[k,0]-cororadius[k]
    xhi=corocenfloor[k,0]+cororadius[k]+1
    ylo=corocenfloor[k,1]-cororadius[k]
    yhi=corocenfloor[k,1]+cororadius[k]+1
    
    #Replace with maximum value from R or C after 
    directdata[ylo:yhi,xlo:xhi]=np.maximum(coro_scaled_gr150cdata[ylo:yhi,xlo:xhi],coro_scaled_gr150rdata[ylo:yhi,xlo:xhi])

#Add information about inputs to output file headers
directpriheader.append(('', ''),end=True)
directpriheader.append(('', 'Grism Flat Patching Information'),end=True)
directpriheader.append(('', ''),end=True)
directpriheader.append(('IFLAT',os.path.basename(directflatfile),'Direct image flat'),end=True)
directpriheader.append(('CFLAT',os.path.basename(gr150cflatfile),'GR150C flat for spots'),end=True)
directpriheader.append(('RFLAT',os.path.basename(gr150rflatfile),'GR150R flat for spots'),end=True)
directpriheader.append(('POM',os.path.basename(pomfile),'POM transmission'),end=True)

#Output files in reference file format
hdup=fits.PrimaryHDU(data=None,header=directpriheader)
hdusci=fits.ImageHDU(data=directdata,header=directsciheader,name='SCI')
hduerr=fits.ImageHDU(data=directdata/200.0,header=hdulist1['ERR'].header,name='ERR')
hdudq=hdulist1['DQ']
hdudq.data[:,:]=0

#Define DQ_DEF binary table HDU
flagtable=np.rec.array([
           ( 0,        1, 'DO_NOT_USE',      'Bad pixel not to be used for science or calibration'   )],
           formats='int32,int32,a40,a80',
           names='Bit,Value,Name,Description')
hdudqdef = fits.BinTableHDU(flagtable,name='DQ_DEF  ',ver=1)
hdulistout=fits.HDUList([hdup,hdusci,hduerr,hdudq,hdudqdef])
print ('Writing',outfile)
hdulistout.writeto(outfile,overwrite=True)
