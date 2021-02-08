# jwst
## Tools for processing and analyzing JWST data
### Additional modules for pipeline processing
 
<b>columnjump</b> is an additional step that can be applied as part of the JWST DETECTOR1 pipeline for data from the NIRISS instrument. The step should be called after dark current subraction and before jump detection. The columnjump step removes random jumps in the levels of some columns (~50 columns per Ng=100 NIRSRAPID ramp) that cause increased noise along those columns. Note the term columns here refers to the original detector coordinates and these are actually rows in the DMD orientation, i.e. they are orthogonal and distinct from the well-known 1/f noise.

A typical calling sequence is:
from columnjump import  ColumnJumpStep  
columnjump = ColumnJumpStep()  
\# Manually set any desired non-default parameter values  
columnjump.nsigma1jump  = 5.00    \# sigma rejection threshold for one jump in the ramp  
columnjump.nsigma2jumps = 3.00    \# sigma rejection threshold for two jumps in the ramp  
columnjump.outputdiagnostics = True    output table of columns corrected?  
columnjump.output_dir = out_dir  
columnjump.output_file = out_file  
\# Run the pipeline  
result = columnjump(inputfile)


### Scripts for analyzing dark exposures 

<b>makebpm.py</b> generates three bad pixel mask files for the NIRISS detector. 
The three masks have different dark noise thresholds appropriate to various types of calibration or science data.
The routine is robust to high rates of cosmic rays and separates cosmic ray hits from noisy pixels. makebpm.py calls makedarknoisefilesgdq.py.

<b>makedarknoisefilesgdq.py</b> makes dark current and noise images from darks, optionally using GDQ flags to mark locations of cosmic ray jumps.

<b>getipc.py</b> determines a 5x5 IPC kernel image based on spread of charge from hot pixels.

<b>makenirissimagingflats.py</b> generates NIRISS imaging flat fields for all filters from one or more integrations per filter that have passed through level 1 pipeline processing.

<b>makenirissgrismflats.py</b> takes in direct (for all filters) and dispersed (currently only F115W and F150W are required) NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror (POM), including the coronagraphic spots. The outputs per filter are 
1. An oversized image of POM transmission including the measured POM outline and features of low transmission. 
2. Grism flat field reference files. These grism flats are equal to the imaging flats over most of the detector, but with all the POM features corrected leaving only the detector response.

For both types, files are generated for each of the GR150C and GR150R grisms. The contents of the files are identical, but separate copies are required for automatic CRDS selection for use with data from each grism.
 
