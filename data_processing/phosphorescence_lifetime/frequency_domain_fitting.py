# -*- coding: utf-8 -*-
"""
Created on Sat Oct 05 13:31:37 2013

@author: Andrew Mark
"""

import numpy as np
from scipy.optimize import leastsq

class SinFit(object):
    """ 
    Fit an AC input signal to a sum of sinusoids with known
    frequencies.
    """
    
    def __init__(self, signal, time, freqs, p0):
        self.signal = signal
        self.time = time
        self.freqs = freqs
        self.N = len(freqs)
        self.p0 = p0
        self.computeFit()
        
    def _peval(self,time,p):
        """ 
        Evaluates the sinusoid fit for a set of fit parameters, p.
        Used by the least squares minimization routine.
        """
        A, phi = p[0:self.N], p[self.N:]
        y = np.zeros(len(time))
        for i in range(0,self.N):        
            y += A[i]*np.cos(2.0*np.pi*self.freqs[i]*time+phi[i])
        return y
    
    def _residuals(self, p, y, x):
        return y - self._peval(x, p)
    
    def computeFit(self):
        self.plsq = leastsq(self._residuals, self.p0, 
                            args=(self.signal, self.time))
        # Add pi phase shifts for negative amplitudes
        for i in range(0, self.N):
            if self.plsq[0][i] < 0:
                self.plsq[0][self.N+i] = self.plsq[0][self.N+i] + np.pi
        # Force ampltiudes to be a positive
        self.plsq[0][0:self.N] = np.abs(self.plsq[0][0:self.N])
    
    def getFit(self, time_vector):
        """ 
        Evaluates the computed fit for a given time vector. 
        """
        return self._peval(time_vector,self.plsq[0])
    
    def getFitParameters(self):
        return self.plsq[0]

class SpectrumFit(object):
    """ 
    Fits amplitudes and phases to the fourier spectrum for a decaying 
    exponential in time.
    """
    
    def __init__(self,freqs, A, phi, Ap0, Pp0):
        self.f = freqs
        self.A = A
        self.phi = phi
        self.Ap0 = Ap0
        self.Pp0 = Pp0
        self.computeFits()
        
    def _pevalAmp(self,f,p):
        """ 
        Evaluates the amplitude spectrum fit for a set of fit 
        parameters, p, at the frequencies in f. 
        """
        scale, tau, offset = p
        return scale/np.sqrt(1.0+(2.0*np.pi*f)**2*tau**2.0)+offset

    def _pevalPhase(self, f, p):
        """
        Evaluates the phase spectrum fit for a set of fit parameters,
        p at the frequencies in f. 
        """
        scale, tau, offset = p
        return scale*np.arctan(-2.0*np.pi*f*tau)+offset
    
    def _residualsAmp(self, p, y, x): return y - self._pevalAmp(x, p)
    
    def _residualsPhase(self, p, y, x): return y - self._pevalPhase(x, p)

    def computeFits(self):
        self.plsqA = leastsq(self._residualsAmp, self.Ap0, 
                             args=(self.A, self.f))
        self.plsqP = leastsq(self._residualsPhase, self.Pp0,
                             args=(self.phi, self.f))
    
    def getAmpFit(self, vector):
        """ 
        Evaluates the computed fit for a given input vector. 
        """
        return self._pevalAmp(vector, self.plsqA[0])
    
    def getPhaseFit(self, vector):
        return self._pevalPhase(vector, self.plsqP[0])

    def getAfitParameters(self): return self.plsqA[0]

    def getPfitParameters(self): return self.plsqP[0]
