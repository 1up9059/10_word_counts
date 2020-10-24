import numpy as np 
import matplotlib.pyplot as plt 
import sys
import json
import pdb
from scipy.optimize import curve_fit 
from lmfit import Model


from matplotlib import style
style.use('fast')


def equation_fit(x, beta, alpha):
    
    y = 1/(pow(x + beta,alpha))
    return y 

def ln_equation_fit(x, beta, alpha):
    
    F = -alpha*np.log(x + beta)
    return F 

        
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

file = str(sys.argv[1])
with open(file, 'r') as dict_file:
    dict_text = json.load(dict_file)


rank = np.asarray(dict_text["Rank"])
frequency = np.asarray(dict_text["Frequency"])

[Final_Data_x, Final_Data_y] = averaging(rank, frequency)

#FIT SCIPY---------------------------------------------------------------------------------------------

#popt, pcov = curve_fit(equation_fit, Final_Data_x, Final_Data_y)
#beta_fit = popt[0]
#alpha_fit = popt[1]

#x_fit = np.linspace(Final_Data_x[0], Final_Data_x[-1], len(Final_Data_x))
#y_fit = equation_fit(x_fit, beta_fit, alpha_fit)

#pdb.set_trace()

#FIT NORMAL LMFT---------------------------------------------------------------------------------------
#pdb.set_trace()
#Fijamos el modelo a usar
model = Model(equation_fit)
#Damos la semilla para el calculo de los parametros
params = model.make_params( beta=2.5, alpha=1)


params['beta'].min = 0.0000000001
params['alpha'].min = 0.0000000001

#Resolvemos el modelo para los params
result = model.fit(Final_Data_y, params, x=Final_Data_x)
#Esto imprime resultados
print("Fit")
print(result.fit_report())

#Capturamos los valores de los parametros de ajuste
params_fit = result.best_values
#beta_fit = -2.73
#A_fit = params_fit['A']
beta_fit = params_fit['beta']
alpha_fit = params_fit['alpha']

#generamos la curva de ajuste
#frequency_fit = result.best_fit
x_fit = np.linspace(Final_Data_x[0], Final_Data_x[-1], len(Final_Data_x))#Entrada de datos para la curva del fit
frequency_fit = equation_fit(Final_Data_x, beta_fit, alpha_fit)#Fit hecho a mano
f_solved = result.best_fit#Fit hecho por el programa
x_solved = np.linspace(f_solved[0], f_solved[-1], len(f_solved))
#pdb.set_trace()



fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
#ax1.plot(rank,  frequency, '-', color ='darkorange', linewidth=4)
#ax1.plot(x_fit, y_fit, linestyle= '-', color='orangered', linewidth = 4 )
ax1.scatter(rank,frequency)
ax1.scatter(Final_Data_x, Final_Data_y)
ax1.scatter(Final_Data_x, frequency_fit)
ax1.scatter(Final_Data_x, f_solved)
#ax1.plot(t2, result2.best_fit, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.plot(t3, result3.best_fit, linestyle= '-', color='orangered', linewidth = 4 )

ax1.set_xlabel("Rank", fontsize=20)
ax1.set_ylabel("Frequency", fontsize=20)
plt.show()