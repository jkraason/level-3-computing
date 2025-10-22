import numpy as np
import matplotlib.pyplot as plt

def plotcoords():
    xvals = np.array([])
    yvals = np.array([])
    file = 'Coordinates.dat'
    x,y = np.loadtxt(file, unpack=True)
    xvals = np.append(xvals,np.array(x))
    yvals = np.append(yvals,np.array(y))
    print(xvals,yvals)
    plt.scatter(xvals,yvals)
    plt.show()

plotcoords()
