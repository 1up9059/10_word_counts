import numpy as np 
import matplotlib.pyplot as plt 
import sys
import json
from lmfit import Model
import pdb

from matplotlib import style
style.use('fast')


def equation_fit(x, beta, alpha):
    
    y = 1/(pow(x + beta,alpha))
    return y 
    

file = str(sys.argv[1])
with open(file, 'r') as dict_file:
    dict_text = json.load(dict_file)

#pdb.set_trace()

#Cargamos ambos contendos desde el json, ambas son listas de floats convertinas a np.arrays
rank = np.asarray(dict_text["Rank"])
frequency = np.asarray(dict_text["Frequency"])

#FIT NORMAL---------------------------------------------------------------------------------------
#pdb.set_trace()
#Fijamos el modelo a usar
model = Model(equation_fit)
#Damos la semilla para el calculo de los parametros
params = model.make_params( beta=2.7, alpha=1.16)

#Establecemos un limite inferior para evitar que el ajuste nos de NaN values
#Si se quita, da NaN's por eso es necesario dejarlo
params['beta'].min = 0.0000000001
params['alpha'].min = 0.0000000001

#Aqui establecemos limistes maximos, pero no es necesario
#params['A'].max = 20
#params['beta'].max = 20
#params['alpha'].max = 20

#Hacemos el ajuste con los parametros y los datos
result = model.fit(frequency, params, x=rank)

#Esto imprime resultados
print("Fit")
print(result.fit_report())

#Capturamos los valores de los parametros de ajuste
params_fit = result.best_values
#beta_fit = -2.73
#A_fit = params_fit['A']
#beta_fit = 1.65*params_fit['beta']
beta_fit = params_fit['beta']
alpha_fit = params_fit['alpha']

#generamos la curva de ajuste
#frequency_fit = result.best_fit
x_fit = np.linspace(rank[0], rank[-1], len(rank))
#frequency_fit = equation_fit(x_fit, A_fit, beta_fit, alpha_fit)
frequency_fit = equation_fit(x_fit, beta_fit, alpha_fit)
f = result.best_fit
#pdb.set_trace()

#MINIMAZER----------------------------------------------------------------


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(rank,  frequency, '-', color ='darkorange', linewidth=4)
ax1.plot(rank, frequency_fit, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.plot(t2, result2.best_fit, linestyle= '-', color='orangered', linewidth = 4 )
#ax1.plot(t3, result3.best_fit, linestyle= '-', color='orangered', linewidth = 4 )

ax1.set_xlabel("Rank", fontsize=20)
ax1.set_ylabel("Frequency", fontsize=20)
plt.show()