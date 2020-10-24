import numpy as np   
from scipy import interpolate
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, UnivariateSpline, interp1d
import matplotlib.pyplot as plt 
plt.style.use('ggplot') 
import pdb 
import sys
import csv

t = []
f = []
sd = []

with open(sys.argv[1]) as csvfile:
    spamreader = csv.reader(csvfile, delimiter= ' ')
    for row in spamreader:
        t.append(float(row[0]))
        f.append(float(row[1]))
        sd.append(float(row[2]))

ti = np.linspace(t[0],t[-1],101)

#Usando el metodo fitpack2
ius = InterpolatedUnivariateSpline(t,f)
#ius.set_smoothing_factor(2)#smooth de la interpolacion
fi1 = ius(ti)

#Usando metodo RBF
rbf = Rbf(t,f)
fi2 = rbf(ti)
#rbf = Rbf(t,f, function='inverse', epsilon=1)
#functions = multiquadric
#            inverse (1/multiquadric)
#            gaussian
#            linear
#            cubic
#            quintic
#            thin_plate


#Usando UnivariateSpline
spl = UnivariateSpline(t, f)
spl.set_smoothing_factor(1)#Regula que tan afinado es la interpolacion
                           #Con cero es lo mismo que InterpolateUnivariateSpline
fi3 = spl(ti)

#usando p1d
pld = interp1d(t, f, kind='cubic')
fi4 = pld(ti)


plt.figure()
plt.plot(t,f, 'bo')
plt.plot(ti,fi1, 'g')
plt.legend(['datos','interpolacion'])
plt.title("Interpolation using InterpolateUnivariateSpline")

plt.figure()
plt.plot(t,f, 'bo')
plt.plot(ti, fi2, 'g')
plt.title("'Interpolation using RBF - multiquadrics'")
plt.legend(['datos','interpolacion'])

plt.figure()
plt.plot(t,f, 'bo')
plt.plot(ti, fi3, 'g')
plt.legend(['datos','interpolacion'])
plt.title("Interpolation using univariatespline")

plt.figure()
plt.plot(t,f, 'bo')
plt.plot(ti, fi4, 'g')
plt.legend(['datos','interpolacion'])
plt.title("Interpolation using p1d")


plt.show()