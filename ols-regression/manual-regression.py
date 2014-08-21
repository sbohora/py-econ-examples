#simple regression with manual ols computation

import pandas as pd
import numpy as np
from numpy.linalg import inv

#load data
data = pd.read_csv('card.csv')

#create variables
data['constant'] = 1 #add constant term to regression
data['exper2'] = data['exper']**2

#prepare data matrices (manually compute regression)
y = np.matrix(np.log(data['wage'])).T #transposed to get column
x = np.matrix(data[['constant','educ','exper','exper2','south','black']])

#OLS: regressing wage on x-variables
beta = inv(x.T * x)*x.T*y
print beta
