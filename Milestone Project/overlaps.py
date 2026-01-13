import numpy as np
import matplotlib.pyplot as plt
def find_overlaps(x, y, r):
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(r)

    N = len(x)
    overlaps = []

    # Handle single radius case
    if r.ndim == 0:
        r = np.full(N, r)

    for i in range(N):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[i] + r[i+1:]
        overlapping = np.where(dist < r_sum)[0]
        for j in overlapping:
            overlaps.append((i, i + 1 + j))

    return overlaps
# Example data
file = 'Coordinates.dat'
xvals = []
yvals = []
x,y = np.loadtxt(file, unpack=True)
xvals = np.append(xvals,np.array(x))
yvals = np.append(yvals,np.array(y))
r = 0.05  # all same radius

overlaps = find_overlaps(xvals, yvals, r)
print("Overlapping pairs:", overlaps)
points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
radius = 0.05
points_radius = 2 * radius / 1.0 * points_whole_ax
plt.scatter(xvals,yvals,s=points_radius**2)
plt.show()