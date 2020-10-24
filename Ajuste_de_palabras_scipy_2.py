import numpy as np 
#import matplotlib.pyplot as plt 
import sys
import json
from scipy.optimize import curve_fit 
from scipy.optimize import minimize
from scipy import stats
import pylab as py
import pdb

from matplotlib import style
style.use('fast')



file = str(sys.argv[1])
with open(file, 'r') as dict_file:
    dict_text = json.load(dict_file)

#Cargamos ambos contendos desde el json, ambas son listas de floats convertinas a np.arrays
rank = np.asarray(dict_text["Rank"])
frequency = np.asarray(dict_text["Frequency"])


def equation_fit(params):
    
    beta = params[0]
    alpha = params[1]
    sd = params[2]
    
    ypre = 1/(pow(rank + beta, alpha))
    #y = -alpha*np.log(x) + beta
    
    y = -np.sum( stats.norm.logpdf(frequency, loc=ypre, scale=sd ) )
    
    return y 
    
initParams = [1, 1, 1]

results = minimize(equation_fit, initParams, method = 'Nelder-Mead')
print(results.x)
estParms = results.x

yout = ypre = 1/(pow(rank + estParms[0], estParms[1]))

print('beta = ' + str(estParms[0]))
print('alpha = ' + str(estParms[1]))

#pdb.set_trace()

py.clf()
py.plot(rank, frequency, 'go')
py.plot(rank, yout)
py.show()


#Creamos el fit con curve_fit

#popt, pcov = curve_fit(equation_fit, rank, frequency)
#popt me da los valores de los parametros del ajuste

#pdb.set_trace()
#A_fit = popt[0]
#beta_fit = popt[0]
#alpha_fit = popt[1]