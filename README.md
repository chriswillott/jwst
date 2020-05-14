# jwst
Tools for processing and analyzing JWST data

makebpm.py generates three bad pixel mask files for the NIRISS detector. 
The three masks have different dark noise thresholds appropriate to various types of calibration or science data.
The routine is robust to high rates of cosmic rays and separates cosmic ray hits from noisy pixels. makebpm.py calls makedarknoisefilesgdq.py.

makedarknoisefilesgdq.py makes dark current and noise images from darks, optionally using GDQ flags to mark locations of cosmic ray jumps.

getipc.py determines a 5x5 IPC kernel image based on spread of charge from hot pixels.

fitpomdefects.py takes in direct and dispersed NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror, including the coronagraphic spots. The two outputs are a mask map of the features and an image of transmission due to the features. The detection direct image flat is usually F115W because this is best to detect the features. Due to the wavelength-dependence of flux loss the filter of the measure direct image flat is usually the output filter.in direct and dispersed NIRISS WFSS flat field images to identify features of low transmission on the NIRISS pick-off mirror, including the coronagraphic spots. The outputs are maps of the features and images of intensity decrease due to the features. 

patchgrismflats.py generates NIRISS WFSS grism flat field by correcting the imaging flat with the POM transmission file.
Coronagraphic spots have too low S/N in POM transmission image so will replace with regions from locally normalized grism flats. Use the F150W grism flats that separate out the dispersed spectra and direct images of sources for all filters.
