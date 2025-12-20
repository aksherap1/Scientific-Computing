#------------------------------------------------------------------------------
# This module implements a numerical integrator for scattered (non-uniform)
# data in one dimension.
#------------------------------------------------------------------------------
import numpy as np
def scattered_trapz(x, y):
"""Numerical integral of scattered data.
Return approximate integral y(x)dx evaluated from tabulated (x,y) points
and estimated approximation error.
Parameters
----------
x, y : 1d arrays, numeric, real, finite
Tabulated, scattered data from underlying (unknown) y=f(x) function.
The array x must be strictly monotonic, either increasing or
decreasing. The values in y are not restricted, but note that values
implying sharp discontinuities in f(x) may result in a poor
approximation of the integral.
Returns
-------
(I, E) : tuple
I : scalar, real
Numerical approximation to the integral
E : scalar, real
Estimated approximation error
Algorithm
---------
An approximation S_full is taken using the trapezoid rule on the
scattered data. A second approximation S_half is taken using only half the
data. Since the trapezoid rule has O(1/N^2) convergence, we guess that
error(S_half) = 4*error(S_full),
from which we can solve the equation:
S_full + E_full = S_half + 4*E_full
and use the extrapolated approximation, I = S_full + E_full as our answer.
"""
S_full = np.sum((y[:-1] + y[1:]) * (x[1:] - x[:-1]) / 2)
x_half = x[::2]
y_half = y[::2]
if x_half[-1] != x[-1]:
x_half = np.append(x_half, x[-1])
y_half = np.append(y_half, y[-1])
S_half = np.sum((y_half[:-1] + y_half[1:]) * (x_half[1:] - x_half[:-1]) / 2)
E = (S_full - S_half) / 3
return (S_full + E, np.abs(E))
### Your work is finished. Below are functions and statements we use to test
### your solution. Feel free to look, but know that if you make any changes
### here they will just be overwritten by our grader. The assignment
### description explains how the submitted script will be called and what the
### expected results should be. That is all you need to know.
### Tests of data integrator ###
def humps(x):
"""Useful test function; int from 0 to 1 is 29.858325395498674."""
return (1/((x - 0.3)**2 + 0.01)) + (1/((x - 0.9)**2 + 0.04)) - 6.0
def test_scattered_humps():
for N in [10, 100, 300, 1000]:
x = np.sort(np.random.random(N))
y = humps(x)
I, e = scattered_trapz(x, y)
print(f"With {N} randomly sampled data:")
print(f"I = {I:0.4f} +/- {e:0.6f}")
x = np.linspace(0, 1, 100)
y = humps(x)
I, e = scattered_trapz(x, y)
print(f"With 100 equally spaced data:")
print(f"I = {I:0.4f} +/- {e:0.6f}")
def test_scattered_data():
import glob
# read file names in working directory
mdl_files = sorted(glob.glob('lores_jupiter_rc=0p?.txt'))
NMoI = np.zeros(len(mdl_files))
NMoI_err = np.zeros(len(mdl_files))
RCore = np.array((0.1, 0.2, 0.3, 0.4, 0.5))
# Calculate NMoI for each model
for k in range(len(mdl_files)):
data = np.loadtxt(mdl_files[k])
rvec = data[:,0]/data[0,0] # a vector of (normalized) radii
dvec = data[:,2]/data[-1,2] # a vector of (normalized) densities
I, I_err = scattered_trapz(rvec, -8*np.pi/3*dvec*rvec**4)
M, M_err = scattered_trapz(rvec, -4*np.pi*dvec*rvec**2)
R = rvec[0]
NMoI = I/(M*R**2)
NMoI_err = NMoI*(I_err/I + M_err/M)
print(f"Integrating density profile in {mdl_files[k]}:")
print(f"NMoI = {NMoI:5.3f} +/- {NMoI_err:0.6f}")
if __name__ == '__main__':
print()
print("Testing scatter_trapz integrator on humps function.")
print("(Correct integral = 29.858325395498674)")
test_scattered_humps()
print()
print()
print("Testing scattered_trapz on Jupiter density models.")
test_scattered_data()
