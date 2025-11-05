import numpy as np
import matplotlib_inline as plt

import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
#eta is a random number
#delta is the amplitude of displacement
#moves are accepted/rejected when particles overlap/are separate
#this code is assuming no overlap between particles - purely for generating the moves

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

d = 0.1
L = 1.0

# Random initial positions
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
#plt.scatter(x,y,s=points_radius**2)
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

### GIF ADDITION ###
os.makedirs("frames", exist_ok=True)
frame_index = 0
total_steps = 20000   # number of MC attempts
save_every = 100     # save a frame every N steps
### EQUILIBRATION GIF SETUP ###
os.makedirs("frames_equil", exist_ok=True)
eq_frame_index = 0
save_every_equil = 100     # save a frame every 100 steps


# Perform 10000 MC moves
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
accepted_moves = 0
for step in range(10000):
    x, y, accepted = mc_move(x, y, r, d, L)
    if accepted:
        accepted_moves += 1
    if step % save_every_equil == 0:
        plt.figure()
        plt.scatter(x, y, s=points_radius**2)
        plt.xlim(0, L); plt.ylim(0, L)
        plt.title(f"Equilibration Step {step}")
        plt.savefig(f"frames_equil/eq_{eq_frame_index:04d}.png")
        plt.close()
        eq_frame_index += 1
    if step%1000 == 0 and step!=0:
        #print(step)
        #print(accepted_moves)
        acceptance_ratio = accepted_moves/step
        print("Acceptance ratio (no modification): ", acceptance_ratio)

        #can always modify this so it just does 1 cycle
while acceptance_ratio>0.5 or acceptance_ratio<0.25:
    if acceptance_ratio>0.5:
        accepted_moves=0
        d = d*1.05
        for j in range(0,10000):
            x,y,accepted = mc_move(x,y,r,d,L)
            if accepted:
                accepted_moves+=1
            if j % save_every_equil == 0:
                plt.figure()
                plt.scatter(x, y, s=points_radius**2)
                plt.xlim(0, L); plt.ylim(0, L)
                plt.title(f"Tuning Step {j}")
                plt.savefig(f"frames_equil/eq_{eq_frame_index:04d}.png")
                plt.close()
                eq_frame_index += 1
        acceptance_ratio = accepted_moves/10000
        print("Acceptance ratio (large):", acceptance_ratio)
    if acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*0.95
        for j in range(0,10000):
            x,y,accepted = mc_move(x,y,r,d,L)
            if accepted:
                accepted_moves+=1
            if j % save_every_equil == 0:
                plt.figure()
                plt.scatter(x, y, s=points_radius**2)
                plt.xlim(0, L); plt.ylim(0, L)
                plt.title(f"Tuning Step {j}")
                plt.savefig(f"frames_equil/eq_{eq_frame_index:04d}.png")
                plt.close()
                eq_frame_index += 1
        acceptance_ratio = accepted_moves/10000
        print("Acceptance ratio (small):", acceptance_ratio)
print("final acceptance ratio:", acceptance_ratio)

N = len(x)
dr = 0.1*r

accepted_moves = 0
for step in range(total_steps):

    x, y, accepted = mc_move(x, y, r, d, L)

    if accepted:
        accepted_moves += 1

    # Save a frame occasionally
    if step % save_every == 0:
        plt.figure()
        plt.scatter(x, y, s=points_radius**2)
        plt.xlim(0, L)
        plt.ylim(0, L)
        plt.title(f"Step {step}")
        plt.savefig(f"frames/frame_{frame_index:04d}.png")
        plt.close()
        frame_index += 1
acceptance_ratio = accepted_moves/step
print("final acceptance ratio:", acceptance_ratio)

#pair correlation function
def pairCorrelationFunction_2D(x, y, L, rMax, dr):
    rho = N/L**2
    n_ideal = rho*2*np.pi*r*dr
    g_r = np.array([])
    num = np.array([])
    r_array = np.arange(0,rMax,dr)
    overlaps = find_overlaps(x, y, r)
    distance = np.zeros((len(x),len(y)))
    for i in (0,len(x)-1):
        for j in (0,len(y)-1):
            if (j!=i) & (j>i):
                x_temp = x[i]-x[j]
                y_temp = y[i]-y[j]
                radius = np.sqrt(y_temp**2+x_temp**2)
            else:
                radius = 0
            distance[i][j] = radius
        # for i in range(0,len(r_array)):
        #     if r_array[i]<2*r:
        #         g_r = np.append(0)
    for k in range(0,len(r_array)-1):
        r_vals = np.where()#between r and r+dr
        num = np.append(len(r_vals),num)
        g_r = np.append(num[k]/n_ideal, g_r)
        #then rerun mc simulations and find the average for g_r
    return g_r, r_vals

g_r,r_vals = pairCorrelationFunction_2D(x,y,L,L/2,dr)
plt.plot(r_vals,g_r)
plt.show()
            


    

### GIF BUILDING ###
frames = []
files = sorted(glob.glob("frames/*.png"))

for f in files:
    frames.append(Image.open(f))

frames[0].save(
    "simulation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=200,
    loop=0
)

print("GIF saved as simulation.gif")


#Equilibration Gif
eq_frames = []
files = sorted(glob.glob("frames_equil/*.png"))

for f in files:
    eq_frames.append(Image.open(f))

eq_frames[0].save(
    "equilibration.gif",
    save_all=True,
    append_images=eq_frames[1:],
    duration=200,
    loop=0
)



print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
#plt.scatter(x,y,s=points_radius**2)
#plt.show()