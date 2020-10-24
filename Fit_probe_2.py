import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
from lmfit import Model
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, UnivariateSpline, interp1d
from scipy.optimize import curve_fit 

import pdb
import sys
import json

def equation_fit(x, beta, alpha):
    
    y = 1/(pow(x + beta,alpha))
    return y 

def ln_equation_fit(x, beta, alpha):
    
    F = -alpha*np.log(x + beta)
    return F 

def R_square(popt, x_data, y_data):
    beta = popt[0]
    alpha = popt[1]
    residuals = y_data - equation_fit(x_data, beta, alpha)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data-np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return r_squared


def normalizer(list1):
    max_val = max(list1)
    normalized_list = []
    for i in list1:
        val = i/max_val
        normalized_list.append(val)
    
    normalized_list = np.asarray(normalized_list)
    return normalized_list

def averaging(x, y):
    
    Final_Data_x = []
    Final_Data_y = []
    
    count_x = x[0]
    count_y = y[0]
    j = 1#lleva el conteo de elementos sumados a count
    
    for i in range(len(x)):
        if i >= 1:
            if y[i-1] == y[i]:
                count_x += x[i]
                count_y += y[i]
                j +=1    
            
            elif y[i-1] != y[i]:
                average_x = count_x/j
                average_y = count_y/j
                Final_Data_x.append(average_x)
                Final_Data_y.append(average_y)
                count_x = x[i]
                count_y = y[i]
                j = 1
            #average_x = 0
        
        #pdb.set_trace()
        
    return [Final_Data_x, Final_Data_y] 


#Carga de los datos
file = str(sys.argv[1])
with open(file, 'r') as dict_file:
    dict_text = json.load(dict_file)
    
#values = dict_text.values()

#Correccion de los datos IGNORE ESTO, NO ES NECESARIO
dict_text["cimar"] = 106
dict_text["cases"] = 11
dict_text["ecosystems"] = 8
dict_text["authorship"] = 4
dict_text["case"] = 3
dict_text["article"] = 5


frequency = [i for i in dict_text.values()]
rank = [i+1 for i in range(len(frequency))]
#pdb.set_trace()


#Tratamiento de los datos
frequecy = np.asarray(frequency)
frequency_normalized = normalizer(frequecy)
rank = np.asarray(rank)
[average_rank, average_f] = averaging(rank, frequency_normalized)

#pdb.set_trace()
#Rank_i = np.linspace(average_rank[0],average_rank[-1],len(average_rank))

#Interpolacion------------------------------------------------------------------------------
#interpolacion usando fitpack2
#interpolation = InterpolatedUnivariateSpline(average_rank, average_f)
#Frequency_i = interpolation(Rank_i)
#[average_Rank, average_F] = averaging(Rank_i, Frequency_i)

#usando p1d
#pld = interp1d(average_rank, average_f, kind='cubic')
#Frequency_i4 = pld(Rank_i)
#[average_Rank, average_F4] = averaging(Rank_i, Frequency_i4)

#pdb.set_trace()
#----------------------------------------------------------------------------------------------------------

popt, pcov = curve_fit(equation_fit, rank, frequency_normalized)
#popt me da los valores de los parametros del ajuste

#pdb.set_trace()
#A_fit = popt[0]
beta_fit = popt[0]
alpha_fit = popt[1]
R_square = R_square(popt, rank, frequency_normalized)
print('beta value = ' + str(beta_fit))
print('alpha = ' + str(alpha_fit))

#pdb.set_trace()

#creamos un set de valores de x para hacer la curva del fit
x_fit = [i for i in range(len(rank))]
y_fit = equation_fit(x_fit, beta_fit, alpha_fit)
#pdb.set_trace()


#Ploteo
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#ax1.set_yscale('log')
#ax1.set_xscale('log')
ax1.grid()
ax1.plot(rank,  frequency_normalized, 'o', color ='darkred',  linewidth=4, markersize=4, label='Data')
#ax1.plot(average_rank, average_f, '-', color ='darkorange', linewidth=4)
ax1.plot(x_fit, y_fit, '-', color ='green', linewidth=4, label='fit')
plt.text(x = 200, y = 0.3, s = u'beta = ' + str(popt[0]) + ', alpha = ' + str(popt[1]) + ", \n" + " R^2 = " + str(R_square), fontsize = 8)
#ax1.plot(average_Rank, average_F, '-', color ='blue', linewidth=4)
#ax1.plot(average_Rank, average_F4, '-', color ='black', linewidth=4)
ax1.set_xlabel("Rank", fontsize=20)
ax1.set_ylabel("Frequency", fontsize=20)
ax1.legend()
plt.show()