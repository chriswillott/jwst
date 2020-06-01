# jwst
Tools for processing and analyzing JWST data

<b>makebpm.py</b> generates three bad pixel mask files for the NIRISS detector. 
The three masks have different dark noise thresholds appropriate to various types of calibration or science data.
The routine is robust to high rates of cosmic rays and separates cosmic ray hits from noisy pixels. makebpm.py calls makedarknoisefilesgdq.py.

<b>makedarknoisefilesgdq.py</b> makes dark current and noise images from darks, optionally using GDQ flags to mark locations of cosmic ray jumps.

<b>getipc.py</b> determines a 5x5 IPC kernel image based on spread of charge from hot pixels.

<b>makenirissimagingflats.py</b> generates NIRISS imaging flat fields for all filters from one or more integrations per filter that have passed through level 1 pipeline processing.


<b>makenirissgrismflats.py</b> takes in direct (for all filters) and dispersed (cuurently only F115W and F150W are required) NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror (POM), including the coronagraphic spots. The outputs per filter are an image of POM transmission due to the features and grism flat field reference files. These grism flats are equal to the imaging flats over most of the detector, but with all the POM features corrected leaving only the detector response.

