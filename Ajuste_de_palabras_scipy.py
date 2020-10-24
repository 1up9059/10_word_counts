import numpy as np 
import matplotlib.pyplot as plt 
import sys
import json
from scipy.optimize import curve_fit 
import pdb

from matplotlib import style
style.use('fast')


def equation_fit(x, beta, alpha):
    
    y = 1/(pow(x + beta, alpha))
    #y = -alpha*np.log(x) + beta
    return y 
    

#def Normalizer(x):
#    x_max = max(x)
#    x_normal = []
#    for i in x:
#        x_normal.append(i/x_max)
#    return x_normal
    

file = str(sys.argv[1])
with open(file, 'r') as dict_file:
    dict_text = json.load(dict_file)

#pdb.set_trace()

#Cargamos ambos contendos desde el json, ambas son listas de floats convertinas a np.arrays
rank = np.asarray(dict_text["Rank"])
frequency = np.asarray(dict_text["Frequency"])

#Creamos el fit con curve_fit

#popt, pcov = curve_fit(equation_fit, rank, frequency)
#popt me da los valores de los parametros del ajuste

pdb.set_trace()
#A_fit = popt[0]
#alpha_fit = popt[1]
#pdb.set_trace()

#creamos un set de valores de x para hacer la curva del fit
#x_fit = np.linspace(rank[0], rank[-1], len(rank))
#y_fit = equation_fit(x_fit, beta_fit, alpha_fit)
#pdb.set_trace()



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(rank,  frequency, '-', color ='darkorange', linewidth=4)
#ax1.plot(x_fit, y_fit, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.scatter(rank,  frequency)
#ax1.scatter(x_fit, y_fit)
#ax1.plot(t2, result2.best_fit, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.plot(t3, result3.best_fit, linestyle= '-', color='orangered', linewidth = 4 )

ax1.set_xlabel("Rank", fontsize=20)
ax1.set_ylabel("Frequency", fontsize=20)
plt.show()