import numpy as np 
import matplotlib.pyplot as plt 
import sys
import json
from scipy.optimize import curve_fit 
import pdb

from matplotlib import style
style.use('fast')


def equation_fit(x, beta, alpha):
    
    y = 1/((x + beta)**alpha)
    #y = -alpha*np.log(x) + beta
    return y 

x_data = np.linspace(0.1, 50, 50)

y = equation_fit(x_data, 2.73, 1.16)#pasamos los x_data para generar los datos de y
np.random.seed(0)#generamos los mismos valores aleatorios en cada ejecucion
y_data_noise = 0.5*np.random.normal(size=x_data.size)#generamos un efecto de ruido
y_data = y + y_data_noise

x_data_log = np.log(x_data)
y_data_log = np.log(y_data)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#ax1.plot(x_data_log,  y_data_log, '-', color ='darkorange', linewidth=4)
ax1.plot(x_data, y, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.plot(t2, result2.best_fit, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.plot(t3, result3.best_fit, linestyle= '-', color='orangered', linewidth = 4 )

ax1.set_xlabel("Rank", fontsize=20)
ax1.set_ylabel("Frequency", fontsize=20)
plt.show()
