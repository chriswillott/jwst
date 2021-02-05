# Columnjump

Performs column jump detection on each ramp integration within a JWST NIRISS exposure to correct for jumps in columns. Note that 'columns' are defined in 
original detector coordinates (not DMS coordinates) so appear in 3rd 
axis of 4D data array. Uses modified version of algorithm documented in CSA-JWST-TN-0003.

It is intended this module be run during the DETECTOR1 pipeline between the dark current subtraction and jump detection step.
