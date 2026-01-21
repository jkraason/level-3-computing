import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def sphere_function(xcentre, ycentre, zcentre,r):
    #draw sphere
    u,v = np.meshgrid(np.linspace(0,np.pi,51),np.linspace(0,2*np.pi, 101))
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    print("x, y, z", x,y,z)
    #shift and scale sphere
    x1 = r*x + xcentre
    y1 = r*y + ycentre
    z1 = r*z + zcentre
    #print(x1,y1,z1)
    return x,y,z

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
xs,ys,zs = sphere_function(1,1,1,1) # same as before
ax.plot_wireframe(xs, ys, zs)
plt.show()