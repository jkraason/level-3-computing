import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
#eta is a random number
#delta is the amplitude of displacement
#moves are accepted/rejected when particles overlap/are separate
#this code is assuming no overlap between particles - purely for generating the moves
import numpy as np

def mc_move(x, y, r, d, L):

    N = len(x)
    r = np.full_like(x, r, dtype=float) if np.ndim(r) == 0 else np.asarray(r)

    # Pick a random particle
    i = np.random.randint(N)
    old_x = x[i]
    old_y = y[i]

    # Save old position
    #if x[i]>=0 and x[i]<=1:
     #   old_x = x[i]
    #else:
     #   old_x = abs(x[i])%1
    #if y[i]>=0 and y[i]<=1:
     #   old_y = y[i]
    #else:
     #   old_y = abs(y[i])%1


    # Propose move
    #dx, dy = np.random.uniform(-delta, delta, size=2)
    eta = random.random()
    x[i] += d*(eta-0.5)
    eta = random.random()
    y[i] += d*(eta-0.5)
    if x[i]>1 or x[i]<0:
        x[i] = abs(x[i])%L
    if y[i]>1 or y[i]<0:
        y[i]=abs(y[i])%L


    # Compute distances to all other particles

    # Ignore self by setting its distance large
        
    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)

    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)
    dist2 = dx_all**2 + dy_all**2
    r_sum2 = 4*(r)**2
    dist2[i] = np.inf
    # Check for overlap
    if np.any(dist2 < r_sum2):
        # Reject move → restore old position
        x[i], y[i] = old_x, old_y
        return x, y, False  # move rejected

    # No overlap → accept move

    return x, y, True

points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
r = 0.03
a = np.pi*r**2
points_radius = 2 * r / 1.0 * points_whole_ax
# Setup
N = 100
d = 0.1
L = 1.0

# Random initial positions (example — not guaranteed non-overlapping)
file = 'Coordinates.dat'
#xvals = np.array([])
#yvals = np.array([])

#x,y = np.loadtxt(file, unpack=True)
#x = np.append(xvals,np.array(x))
#y = np.append(yvals,np.array(y))
# --- lattice initialization ---
Nx = 10   # 10×10 = 100 particles
Ny = 10
x = np.zeros(Nx*Ny)
y = np.zeros(Nx*Ny)

# spacing between particle centers
spacing = 2.2 * r   # > 2r prevents overlap

index = 0
for i in range(Nx):
    for j in range(Ny):
        x[index] = i * spacing
        y[index] = j * spacing
        index += 1

# Normalise to box of size L
x /= max(x)
y /= max(y)
plt.scatter(x,y,s=points_radius**2)
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

# Perform 1000 MC moves
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
accepted_moves = 0
for step in range(1000):
    x, y, accepted = mc_move(x, y, r, d, L)
    if accepted:
        accepted_moves += 1
acceptance_ratio = accepted_moves/1000
print("initial acceptance ratio:", acceptance_ratio)


#checks if the acceptance ratio is in the correct range
while acceptance_ratio>0.5 or acceptance_ratio<0.25:
    if acceptance_ratio>0.5:
        accepted_moves = 0
        d = d*1.05
        #print(d)
        for j in range(0,1000):
            x, y, accepted = mc_move(x, y, r, d, L)
            #print(x,y,accepted)
            if accepted:
                accepted_moves += 1
        acceptance_ratio = accepted_moves/1000
        print("Acceptance ratio (large):", acceptance_ratio)
    if acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*0.95
        #print(d)
        for j in range(0,1000):
            x, y, accepted = mc_move(x, y, r, d, L)
            if accepted:
                accepted_moves += 1
        acceptance_ratio = accepted_moves/1000
print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
plt.scatter(x,y,s=points_radius**2)
plt.show()