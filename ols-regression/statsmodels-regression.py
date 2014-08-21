#simple regression using statsmodels package
#requires statsmodels version 0.5 or newer (comes with Anaconda)

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

#load data (make sure that python is running in the current directory)
obs = pd.read_csv('card.csv')

#create variables
obs['exper2'] = obs['exper']**2
obs['lnwage'] = np.log(obs['wage'])

#regression
model = sm.ols(formula='lnwage ~ educ + exper + exper2 + south + black', data=obs)
regression = model.fit()
print regression.summary()

errors = regression.resid # residuals
