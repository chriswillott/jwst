"""
Find and fix jumps in columns of reference pixels in each 
integration within the input data array. The input data array 
is assumed to be in units of DN. Note that 'columns' are defined in 
original detector coordinates (not DMS coordinates) so appear in 3rd 
axis of 4D data array.
"""

import logging
import numpy as np
import numpy.ma as ma
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy.io import ascii   
import sys
import time

from jwst import datamodels
from jwst.datamodels import dqflags

from .jwstpipe1p1p0_ramp_fit import calc_slope
from .jwstpipe1p1p0_utils import OptRes

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

HUGE_NUM = np.finfo(np.float32).max


def quickrampfit(maskeddatathisint,ngroups):
    """
    Perform quick ramp fit on all reference pixels to determine outliers in slope along each column
    """
    #Calculate the slope of all reference pixels using pipeline ramp_fit routine
    data_sect = maskeddatathisint.data
    gain_sect = (0.0*data_sect)+1.0
    rn_sect = (0.0*data_sect)+1.0
    gdq_sect = (0.0*data_sect).astype(int)
    frame_time = 10.7
    save_opt = True
    opt_res  =  OptRes(1, np.squeeze(data_sect[0,:,:]).shape, 1, ngroups, save_opt)
    t_dq_cube, inv_var, opt_res, f_max_seg, num_seg = calc_slope(data_sect, gdq_sect, frame_time, opt_res, save_opt, rn_sect, gain_sect, 1, ngroups, 'optimal', 1)
    refpixslopes = opt_res.slope_2d.reshape(np.squeeze(data_sect[0,:,:]).shape)
    return refpixslopes


def findcolrefpixsubmean(col,colstart,colstop,ampindex,maskeddatathisint,ngroups,nrows):
    """
    Find clipped mean of a single column of reference pixels and return the reference pixels with 
    the clipped mean subtracted.
    """
    #Get 2.5sigma-clipped mean of the good reference pixels in this column for each group.
    #If a pixel is clipped out then remove it from all groups.
    datathiscol = maskeddatathisint[:,col,:]
    clipmaskeddata = sigma_clip(datathiscol,sigma = 2.5,maxiters=3,axis=1)
    clipmaskeddata = ma.mask_rowcols(clipmaskeddata,axis=1)
    colrefpix = np.mean(clipmaskeddata,axis=1)
    numgoodrefpix = clipmaskeddata.count(axis=1)

    #Get clipped mean of neighouring 20 columns of reference pixels and subtract it off. 
    numcompare = 20             
    colnear = np.arange((numcompare+1),dtype=int)+col-numcompare//2
    #For columns at edges of each amplifier, use the nearest 20 pixels in that amp.
    if (np.min(colnear)<colstart[ampindex]):
        colnear -= (np.min(colnear)-colstart[ampindex])
    if (np.max(colnear)>(colstop[ampindex]-1)):
        colnear -= ((np.max(colnear)-(colstop[ampindex]-1)))
    exclude = np.where(colnear==col)
    colnear = np.delete(colnear, exclude[0])
    datathese20cols = maskeddatathisint[:,colnear,:]
    #Mask any rows that were masked for the single column
    singlemask = np.tile(clipmaskeddata[0,:].mask,(ngroups,numcompare,1))
    datathese20cols = ma.array(datathese20cols.data, mask=singlemask)
    refpixcomparereshape = datathese20cols.reshape((ngroups,(nrows*numcompare)))
    cliprefpixcomparereshape = sigma_clip(refpixcomparereshape,sigma=2.5,maxiters=3,axis=1)
    meanrefpixcomparison = np.mean(cliprefpixcomparereshape,axis=1)

    #Subtract mean of comparison neighbouring columns from mean of this column
    colrefpixsubmean = colrefpix-meanrefpixcomparison
    colrefpixsubmean = colrefpixsubmean.astype('float16')
    return colrefpixsubmean


def findfix_jumps(data, pdq, nsigma1jump, nsigma2jumps, nframes, filename, outputdiagnostics):
    """
    Main function of rampbreaks.py
    """
    #set up output diagnostic file if requested
    if outputdiagnostics == True:
        outtablename = 'rampbreaksout_'+filename+'.dat'
        outtable = Table(names=('Column','Numsigma1break','Numsigma2break','Ngroup1break','Ngroup2break1','Ngroup2break2','Size1break','Size2break1','Size2break2'),dtype=('i4', 'f8', 'f8', 'i4', 'i4', 'i4', 'f8', 'f8', 'f8'))

    #Set up correction array same shape as data
    correctionarray = np.zeros((data.shape),dtype='float16')

    #Make pixel DQ array same shape as data
    pdq = np.repeat(pdq[np.newaxis, :, :], data.shape[1], axis=0)
    pdq = np.repeat(pdq[np.newaxis, :, :, :], data.shape[0], axis=0)

    #Shrink data and pdq arrays to leave only top and bottom reference pixels of active columns
    databottom = data[:,:,4:2044,0:4]
    datatop = data[:,:,4:2044,2044:]
    datarp = np.concatenate((databottom,datatop),axis=3)
    pdqbottom = pdq[:,:,4:2044,0:4]
    pdqtop = pdq[:,:,4:2044,2044:]
    pdqrp = np.concatenate((pdqbottom,pdqtop),axis=3)

    #Mask reference pixels that have DO_NOT_USE flag set
    wh_donotuse = np.where(np.bitwise_and(pdqrp, dqflags.pixel['DO_NOT_USE']))
    pdqrp[:,:,:,:] = 0
    pdqrp[wh_donotuse] = 1

    #Get refpix shrunken data characteristics
    (nints, ngroups, ncols, nrows) = datarp.shape

    #Scale sigma thresholds based on calibration in darks so approx 
    #95% of corrections are good for full range of N_groups and N_frames
    if nframes == 1:
        gradient = 74.0
        scaling = 0.01*(100-gradient)+gradient/ngroups
    elif nframes == 4:
        gradient = 100.0/4.0
        scaling = 0.04*(22-gradient)+gradient/ngroups
    else:
        log.info('NFRAMES={} not valid for NIRISS (!=1 or 4)'.format(nframes))
        log.info('Column jump step will be skipped')
        return
  
    nsigma1jump *= scaling
    nsigma2jumps *= scaling
    log.info('nsigma1jump scaled to %s' % nsigma1jump)
    log.info('nsigma2jumps scaled to %s' % nsigma2jumps)

    #Set up degrees of freedom based on ngroups
    dof = float(ngroups-1-1-1)
    countcoljump = 0

    # Loop over multiple integrations
    for integration in range(nints):
        log.info('working on integration %d' % (integration+1))

        #Make masked array of reference pixel data for this integration only
        datathisint = datarp[integration,:,:,:]
        pdqthisint = pdqrp[integration,:,:,:]
        maskeddatathisint = ma.array(datathisint,mask=pdqthisint)

        #Make quick ramp fit to all reference pixels to identify outliers not in mask and add them to mask 
        refpixslopes = quickrampfit(maskeddatathisint,ngroups)
        maskedrefpixslopes = ma.array(refpixslopes, mask=pdqthisint[0,:,:])
        clipmaskedrefpixslopes = sigma_clip(maskedrefpixslopes,sigma=2.2,maxiters=3,axis=1)

        #Add these bad pixels to mask
        singlemask = np.tile(clipmaskedrefpixslopes.mask,(ngroups,1,1))
        maskeddatathisint = ma.array(maskeddatathisint.data, mask=singlemask)

        #Use per amplifier mean of top-bottom reference pixels noise 
        amplifier = np.array(['A','B','C','D'])
        colstart = np.array([4,512,1024,1536])-4
        colstop = np.array([512,1024,1536,2044])-4
        refpixareavarianceamp = np.zeros(4)
        for k in range(4):
            refpixelsthisamp = maskeddatathisint[:,colstart[k]:colstop[k],:]
            refpixelsnoisethisamp = sigma_clip(np.std(refpixelsthisamp,axis=0),sigma=3,maxiters=3)
            meanrefpixelsnoisethisamp = np.mean(refpixelsnoisethisamp)
            #refpixareanoiseamp is the expected noise for average of 7 reference pixels 
            refpixareanoiseamp = meanrefpixelsnoisethisamp/(7.0**0.5)
            refpixareavarianceamp[k] = refpixareanoiseamp**2.0

        #set up arrays to store first pass results
        onebreaksize = np.zeros(ncols)
        onebreakbestk = np.zeros(ncols, dtype=int)
        onebreakchisqdiff = np.zeros(ncols)
        twobreak1size = np.zeros(ncols)
        twobreak2size = np.zeros(ncols)
        twobreakbestk = np.zeros(ncols, dtype=int)
        twobreakbestm = np.zeros(ncols, dtype=int)
        twobreakchisqdiff = np.zeros(ncols)

        #loop over original detector coordinate system columns
        for col in range(ncols):

            #which amplifier
            ampindex = np.where((col >= colstart)&(col < colstop))
            refpixareavariance = refpixareavarianceamp[ampindex][0]
            
            #get reference pixels values for this column minus a clipped mean of them 
            colrefpixsubmean = findcolrefpixsubmean(col,colstart,colstop,ampindex,maskeddatathisint,ngroups,nrows)

            #calculate chi-square for the flat model with no break
            chisqnobreak = (1.0/dof)*np.sum(((colrefpixsubmean-np.mean(colrefpixsubmean))**2.0)/refpixareavariance)
 
            chisqmin = chisqnobreak 
            chisqmin2break = chisqnobreak 

            #calculate chi-square for a model that has a break at each value of N_group
            chisqnorm = 1.0/(dof*refpixareavariance)
            for k in range(ngroups):
                if k>2 and k<(ngroups-3):
                    lomean = np.mean(colrefpixsubmean[:k])
                    himean = np.mean(colrefpixsubmean[k:])
                    model = np.concatenate(((np.repeat(lomean,(k+1))),(np.repeat(himean,(ngroups-(k+1))))))
                    chisq = chisqnorm*np.sum((colrefpixsubmean-model)**2.0)

                    #if the chi-square is better than any previous model set it as the best
                    if chisq<chisqmin:
                        chisqmin = chisq
                        onebreakbestk[col] = k+1
                        onebreaksize[col] = lomean-himean
                        bestmodel1break = model

            onebreakchisqdiff[col] = chisqnobreak-chisqmin

        #Do statistics to find which chisqdiff are outliers for one break
        cliponebreakchisqdiff = sigma_clip(onebreakchisqdiff,sigma=5.0,maxiters=3)
        limitonebreakchisqdiff = np.ma.max(cliponebreakchisqdiff)
        medianonebreakchisqdiff = np.ma.median(cliponebreakchisqdiff)
        numsigmaonebreak = (onebreakchisqdiff-medianonebreakchisqdiff)/((limitonebreakchisqdiff-medianonebreakchisqdiff)/5.0)

        #Loop over columns again for model with 2 breaks in ramp
        for col in range(ncols):
            # Run on only those that show a high chisqdiff for one break because faster
            if numsigmaonebreak[col]>nsigma1jump:
                #which amplifier
                ampindex = np.where((col >= colstart)&(col < colstop))
                refpixareavariance = refpixareavarianceamp[ampindex][0]

                #get reference pixels values for this column minus a clipped mean of them 
                colrefpixsubmean = findcolrefpixsubmean(col,colstart,colstop,ampindex,maskeddatathisint,ngroups,nrows)

                #calculate chi-square for the flat model with no break
                chisqnobreak = (1.0/dof)*np.sum(((colrefpixsubmean-np.mean(colrefpixsubmean))**2.0)/refpixareavariance)

                chisqmin = chisqnobreak 
                chisqmin2break = chisqnobreak 
                chisqnorm = 1.0/(dof*refpixareavariance)

                #calculate chi-square for a model that has two breaks at different values of N_group
                for k in range(ngroups):
                    if k>2 and k<(ngroups-6):
                        for m in range(ngroups):
                            if m-k>2 and m<(ngroups-3):
                                lomean = np.mean(colrefpixsubmean[:k])
                                midmean = np.mean(colrefpixsubmean[k:m])
                                himean = np.mean(colrefpixsubmean[m:])
                                model2break = np.concatenate(((np.repeat(lomean,(k+1))),(np.repeat(midmean,(m-k))),(np.repeat(himean,(ngroups-(m+1))))))
                                chisq2break = chisqnorm*np.sum((colrefpixsubmean-model2break)**2.0)

                                #if the chi-square is better than any previous model set it as the best
                                if chisq2break<chisqmin2break:
                                    chisqmin2break = chisq2break
                                    twobreakbestk[col] = k+1
                                    twobreakbestm[col] = m+1
                                    twobreak1size[col] = lomean-midmean
                                    twobreak2size[col] = midmean-himean
                                    bestmodel2break = model2break

                twobreakchisqdiff[col] = chisqnobreak-chisqmin2break

        #Do statistics to find when outliers in two break chisqdiff - one break chisqdiff
        twominusonebreakschisqdiff = twobreakchisqdiff-onebreakchisqdiff
        significanttwominusoneb = twominusonebreakschisqdiff[np.where(onebreakchisqdiff>limitonebreakchisqdiff)]
        clipsignificanttwominusoneb = sigma_clip(significanttwominusoneb,sigma=3.0,maxiters=3)
        limittwobreakchisqdiff = np.ma.max(clipsignificanttwominusoneb)
        mediantwobreakchisqdiff = np.ma.median(clipsignificanttwominusoneb)
        numsigmatwobreak = (twominusonebreakschisqdiff-mediantwobreakchisqdiff)/((limittwobreakchisqdiff-mediantwobreakchisqdiff)/3.0)

        #Loop over columns again to determine best model
        for col in range(ncols):
            #Run on only those that show a high chisqdiff for one break
            if numsigmaonebreak[col]>nsigma1jump:
                #Decide on whether to use 1 break or 2 break model
                if numsigmatwobreak[col]>nsigma2jumps:
                    correctionarray[integration,twobreakbestk[col]:,(col+4),:]+=twobreak1size[col]
                    correctionarray[integration,twobreakbestm[col]:,(col+4),:]+=twobreak2size[col]
                    log.info(' column %d: 2 breaks of sizes %.3f,%.3f at groups %d,%d with numsigmatwobreak=%.3f' % ((col+4),twobreak1size[col],twobreak2size[col],twobreakbestk[col],twobreakbestm[col],numsigmatwobreak[col]))
                else:
                    correctionarray[integration,onebreakbestk[col]:,(col+4),:]+=onebreaksize[col]
                    log.info(' column %d: 1 break of size %.3f at group %d with numsigmaonebreak=%.3f' % ((col+4),onebreaksize[col],onebreakbestk[col],numsigmaonebreak[col]))
                    
                countcoljump = countcoljump+1

                #Add to output table - column numbering is zero-indexed for full detector
                if outputdiagnostics == True:
                    outtable.add_row(((col+4),numsigmaonebreak[col],numsigmatwobreak[col],onebreakbestk[col],twobreakbestk[col],twobreakbestm[col],onebreaksize[col],twobreak1size[col],twobreak2size[col]))

        #Finished all column loops
        log.info(' Done: corrected jumps in %d columns in this integration' % (countcoljump))
    
    #Next integration (integration loop)

    #Output diagnostics if requested and at least one jump found
    if countcoljump > 0 and outputdiagnostics == True:
        #pdf_pages.close()
        ascii.write(outtable,outtablename,format='fixed_width_two_line',overwrite=True)

    #Update the data with the corrections before returning
    data+=correctionarray

    return


