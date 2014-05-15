# -*- coding: utf-8 -*-
"""
Created on Sat Oct 05 13:46:23 2013

@author: Andrew Mark
"""

import numpy as np
from matplotlib import pyplot as plt

from frequency_domain_fitting import SinFit, SpectrumFit   

time = np.arange(0,0.02,0.0001)

freqs = np.array([100.0, 300.0, 500.0, 800.0])
A, phi = np.array([0.9, 0.4, 0.6, 0.1]), np.array([0.4, 0.0, 0.8, 0.1])

y_true = np.zeros(len(time))
y_meas = np.zeros(len(time))

# Create test signals
for i in range(0,4):        
    y_true += A[i]*np.cos(2.0*np.pi*freqs[i]*time+phi[i])

for i in range(0,4):
    y_meas += A[i]*np.cos(2.0*np.pi*freqs[i]*time+phi[i])
    y_meas += 0.1*np.random.randn(len(time))

# Create fake inital guess parameters
a0 = np.array([A[0]+0.6, A[1]-0.2, A[2]+0.3, A[3]-0.1])
phi0 = np.array([phi[0]+0.4, phi[1]-0.5, phi[2]+0.6, phi[3]-0.1])
p0 = np.concatenate((a0, phi0))

fit = SinFit(y_meas, time, freqs, p0)

plt.plot(time, y_meas,'k.',time, fit.getFit(time))

print "Parameters of the fit"
print fit.getFitParameters()
print "Actual parameters of the test function"
print np.concatenate((A,phi))

plt.show()
