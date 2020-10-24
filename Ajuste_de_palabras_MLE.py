import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats

import pymc3 as pm3
import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import json

file = str(sys.argv[1])
with open(file, 'r') as dict_file:
    dict_text = json.load(dict_file)
    
    #Cargamos ambos contendos desde el json, ambas son listas de floats convertinas a np.arrays
rank = np.asarray(dict_text["Rank"])
frequency = np.asarray(dict_text["Frequency"])

df = pd.DataFrame({'y':frequency, 'x':rank})
df['constant'] = 2
sns.regplot(df.x, df.y)

plt.show()