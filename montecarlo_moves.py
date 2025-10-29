import numpy as np
import random
import matplotlib.pyplot as plt
#eta is a random number
#delta is the amplitude of displacement
#moves are accepted/rejected when particles overlap/are separate
#this code is assuming no overlap between particles - purely for generating the moves
import numpy as np

def mc_move(x, y, r, d, box=None):

    #box = np.array(Lx,Ly)
    N = len(x)
    r = np.full_like(x, r, dtype=float) if np.ndim(r) == 0 else np.asarray(r)

    # Pick a random particle
    i = np.random.randint(N)
    

    # Save old position
    old_x, old_y = x[i], y[i]

    # Propose move
    #dx, dy = np.random.uniform(-delta, delta, size=2)
    eta = random.random()
    x[i] += d*(eta-0.5)
    eta = random.random()
    y[i] += d*(eta-0.5)

    # Apply periodic boundaries (if box specified)
    if box is not None:
        x[i] = x[i] % box
        y[i] = y[i] % box

    # Compute distances to all other particles
    dx_all = x[i] - x
    dy_all = y[i] - y
    dist2 = dx_all**2 + dy_all**2
    r_sum2 = 4*(r)**2

    # Ignore self by setting its distance large
    dist2[i] = np.inf

    # Check for overlap
    if np.any(dist2 < r_sum2):
        # Reject move → restore old position
        x[i], y[i] = old_x, old_y
        return x, y, False  # move rejected

    # No overlap → accept move
    return x, y, True

points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
r = 0.05
points_radius = 2 * r / 1.0 * points_whole_ax
# Setup

d = 0.9

# Random initial positions (example — not guaranteed non-overlapping)
file = 'Coordinates.dat'
x = np.array([1.0,1.0])
y = np.array([1.0,1.1])

#x,y = np.loadtxt(file, unpack=True)
#x = np.append(xvals,np.array(x))
#y = np.append(yvals,np.array(y))
#plt.scatter(x,y,s=points_radius**2)

# Perform 10,000 MC moves
accepted_moves = 0
for step in range(1000):
    x, y, accepted = mc_move(x, y, r, d)
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
            x, y, accepted = mc_move(x, y, r, d)
            #print(x,y,accepted)
            if accepted:
                accepted_moves += 1
        acceptance_ratio = accepted_moves/1000
        print(f"Acceptance ratio (large):", acceptance_ratio)
    elif acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*1.05
        #print(d)
        for j in range(0,1000):
            x, y, accepted = mc_move(x, y, r, d)
            if accepted:
                accepted_moves += 1
        acceptance_ratio = accepted_moves/1000
        print(f"Acceptance ratio (small):", acceptance_ratio)

plt.scatter(x,y,s=points_radius**2)
plt.show()