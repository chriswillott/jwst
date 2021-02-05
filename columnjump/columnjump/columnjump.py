import time
import logging

from . import rampbreaks as rbreaks

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def detect_columnjumps (input_model, nsigma1jump, nsigma2jumps, outputdiagnostics):
    """
    This is the high-level controlling routine for the column jump detection 
    process. It loads and sets the various input data and parameters needed 
    and then calls the detection method.
    """

    # Load the data arrays that we need from the input model
    output_model = input_model.copy()
    data = input_model.data
    err  = input_model.err
    gdq  = input_model.groupdq
    pdq  = input_model.pixeldq

    ngroups = data.shape[1]
    nframes = input_model.meta.exposure.nframes
    filename = input_model.meta.filename
    filename = filename.replace('.fits','')

    # Apply the column jump detection and correction algorithm
    log.info('Executing column jump detection and correction algorithm')
    start = time.time()

    rbreaks.findfix_jumps(data, pdq, nsigma1jump, nsigma2jumps, nframes, filename, outputdiagnostics)

    elapsed = time.time() - start
    log.debug('Elapsed time = %g sec' %elapsed)

    # Update the arrays of the output model with the jump detection results
    output_model.data    = data
    output_model.groupdq = gdq
    output_model.pixeldq = pdq

    return output_model
