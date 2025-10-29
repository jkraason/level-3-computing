import numpy as np
import matplotlib.pyplot as plt
#plots the coordinates from the document
def plotcoords():
    xvals = np.array([])
    yvals = np.array([])
    file = 'Coordinates.dat'
    x,y = np.loadtxt(file, unpack=True)
    xvals = np.append(xvals,np.array(x))
    yvals = np.append(yvals,np.array(y))
    print(xvals,yvals)
    points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
    radius = 0.05
    points_radius = 2 * radius / 1.0 * points_whole_ax
    plt.scatter(xvals,yvals,s=points_radius**2)
    plt.show()
plotcoords()
