#for putting all of the functions together
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
    radius = 0.045
    points_radius = 2 * radius / 1.0 * points_whole_ax
    plt.scatter(xvals,yvals,s=points_radius**2)
    plt.show()
    return(xvals,yvals)
plotcoords()

def find_overlaps(xvals, yvals, radius):

    N = len(xvals)
    overlaps = []

    # Handle single radius case
    if r.ndim == 0:
        r = np.full(N, radius)

    for i in range(N):
        dx = xvals[i] - xvals[i+1:]
        dy = yvals[i] - yvals[i+1:]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = radius[i] + radius[i+1:]
        overlapping = np.where(dist < r_sum)[0]
        for j in overlapping:
            overlaps.append((i, i + 1 + j))

    return overlaps
find_overlaps(xvals,yvals,radius)