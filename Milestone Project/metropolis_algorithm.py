import numpy as np
import matplotlib.pyplot as plt
import random
xvals = np.array([])
yvals = np.array([])
file = 'Coordinates.dat'
x,y = np.loadtxt(file, unpack=True)
x = np.append(xvals,np.array(x))
y = np.append(yvals,np.array(y))
U_j = random.random()
print(U_j)
U_i = np.linspace(0,1,len(x))
d=0.1
k_b = 1.38E-23
T = 300
plt.scatter(x,y)
for i in range(len(U_i)):
    if U_j < U_i[i]:
        #move accepted
        eta = random.random()
        x[i]= x[i]+d*(eta-0.5)
        eta = random.random()
        y[i]= y[i]+d*(eta-0.5)

    if U_j<U_i[i]:
        #move accepted with probability P = e^(U_i-U_j)/k_bT
        r = random.random()
        p=np.exp((U_i[i]-U_j)/k_b*T)
        if r < p:
            #move accepted
            eta = random.random()
            #print(x[i])
            x[i]=x[i]+d*(eta-0.5)
            #print(x[i])
            eta = random.random()
            y[i]=y[i]+d*(eta-0.5)
        if r > p:
            #move rejected
            x[i]=x[i]
            y[i]=y[i]

plt.scatter(x,y)
plt.show()
