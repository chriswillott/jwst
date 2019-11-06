# jwst
Tools for processing and analyzing JWST data

makebpm.py generates three bad pixel mask files for the NIRISS detector. 
The three masks have different dark noise thresholds appropriate to various types of calibration or science data.
The routine is robust to high rates of cosmic rays and separates cosmic ray hits from noisy pixels. makebpm.py calls makedarknoisefilesgdq.py.

makedarknoisefilesgdq.py makes dark current and noise images from darks, optionally using GDQ flags to mark locations of cosmic ray jumps.

getipc.py determines a 5x5 IPC kernel image based on spread of charge from hot pixels.

