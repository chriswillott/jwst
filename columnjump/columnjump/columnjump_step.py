#! /usr/bin/env python

from jwst.stpipe import Step
from jwst import datamodels
from .columnjump import detect_columnjumps
import time

__all__ = ["ColumnJumpStep"]

class ColumnJumpStep(Step):
    """
    ColumnJumpStep: Performs column jump detection on each ramp integration within an
    exposure to correct for jumps in columns. 
    Uses modified version of algorithm in CSA-JWST-TN-0003. 
    """

    spec = """
        nsigma1jump                = float(default=5.0,min=0)  # sigma rejection threshold for one jump
        nsigma2jumps               = float(default=3.0,min=0)  # sigma rejection threshold for two jumps
        outputdiagnostics          = boolean(default=False)    # output table of columns corrected? 
    """

    #It is assumed that most bad reference pixels are flagged in the PixelDQ array of the data so no mask reference file is required.

    def process(self, input):

        with datamodels.RampModel(input) as input_model:
            tstart = time.time()

            # Check for consistency between keyword values and data shape
            ngroups = input_model.data.shape[1]
            ngroups_kwd = input_model.meta.exposure.ngroups
            if ngroups != ngroups_kwd:
                self.log.error("Keyword 'NGROUPS' value of '{0}' does not match data array size of '{1}'".format(ngroups_kwd,ngroups))
                raise ValueError("Bad data dimensions")

            # Check for an input model with NGROUPS<=6. 
            if ngroups <= 6:
                self.log.warn('Will not apply column jump detection when NGROUPS<=6;')
                self.log.warn('Column jump step will be skipped')
                result = input_model.copy()
                result.meta.cal_step.jump = 'SKIPPED'
                return result

            # Retrieve the parameter values
            nsigma1jump  = self.nsigma1jump
            nsigma2jumps = self.nsigma2jumps
            outputdiagnostics = self.outputdiagnostics
            self.log.info('Sigma rejection threshold for one break = %g ', nsigma1jump)
            self.log.info('Sigma rejection threshold for two breaks = %g ', nsigma2jumps)
            self.log.info('Output Diagnostic files = %s ', outputdiagnostics)

           # Call the column jump detection routine
            result = detect_columnjumps(input_model, nsigma1jump, nsigma2jumps, outputdiagnostics)

            tstop = time.time()
            self.log.info('The execution time in seconds: %f', tstop - tstart)

        result.meta.cal_step.columnjump = 'COMPLETE'

        return result

