#Time series example: generate series, visualize, regression
import numpy as np
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa import ar_model
import matplotlib.pyplot as plt
import pandas as pd

def ar1(phi,n):
    """
    Generates an instance of an AR(1) process with Gaussian white noise
    phi: coefficient
    n: length of series
    Returns: the series
    """
    e = np.random.standard_normal(n)
    y = np.zeros(n)
    for t in np.arange(1,n):
        y[t] = phi*y[t-1] + e[t]
    return y

def ar2(phi1,phi2,n):
    """
    Generates an instance of an AR(2) process with Gaussian white noise
    phi1,phi2: coefficients
    n: length of series
    Returns: the series
    """
    e = np.random.standard_normal(n)
    y = np.zeros(n)
    for t in np.arange(2,n):
        y[t] = phi1*y[t-1] + phi2*y[t-2] + e[t]
    return y


# examples
n = 100
y = ar2(0.7,0.2,n)
ncorr = 25 # number of lags to compute for the autocorrelation functions
y_acf = acf(y,nlags=ncorr)
y_pacf = pacf(y,nlags=ncorr)

# plot the series
fig,ax = plt.subplots(figsize=(14,4))
ax.plot(y,label=r'$y_t$')
ax.legend()
ax.set_title(r'$y_t= \phi_1 y_{t-1} + \phi_2 y_{t-2} + \epsilon_t$')
plt.show()

#plot the acf and pacf
fig2,axes = plt.subplots(2)
fig2.subplots_adjust(hspace=0.5)
axes[0].bar(np.arange(ncorr+1), y_acf)
axes[0].set_title("Autocorrelation")
axes[1].bar(np.arange(ncorr+1), y_pacf)
axes[1].set_title("Partial Autocorrelation")
plt.show()

#organize and print correlogram
cgram = pd.DataFrame([y_acf,y_pacf]).transpose()
cgram.columns = 'acf','pacf'
print 'Correlogram'
print cgram

#regression using AR model
reg_model = ar_model.AR(y)
print '\nBIC selects order {}.'.format(reg_model.select_order(6,'bic'))
reg_results = reg_model.fit(maxlag=6,ic='bic')
#print out results (ar_model doesn't come with a summary function)
print 'Regression results:\nNumber of observations (T-k): {}\nOrder: {}\n'.format(reg_results.nobs,reg_results.k_ar)
print 'coeff: ', reg_results.params
print 'std err: ', reg_results.bse
print 't-stat: ', reg_results.tvalues
print 'p-value: ', reg_results.pvalues
