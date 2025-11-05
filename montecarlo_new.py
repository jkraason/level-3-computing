import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os  # <-- added

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

    # Propose move
    eta = random.random()
    x[i] += d*(eta-0.5)
    eta = random.random()
    y[i] += d*(eta-0.5)

    # Apply periodic boundaries
    if x[i]>1 or x[i]<0:
        x[i] = abs(x[i])%L
    if y[i]>1 or y[i]<0:
        y[i]=abs(y[i])%L

    # Compute PBC distances
    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)

    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)

    dist2 = dx_all**2 + dy_all**2
    r_sum2 = 4*(r)**2
    dist2[i] = np.inf

    # Check for overlap
    if np.any(dist2 < r_sum2):
        x[i], y[i] = old_x, old_y
        return x, y, False

    return x, y, True


points_whole_ax = 5 * 0.8 * 72
r = 0.03
a = np.pi*r**2
points_radius = 2 * r / 1.0 * points_whole_ax

# Setup
N = 100
d = 0.1
L = 1.0

# --- lattice initialization ---
Nx = 10
Ny = 10
x = np.zeros(Nx*Ny)
y = np.zeros(Nx*Ny)

spacing = 2.2 * r

index = 0
for i in range(Nx):
    for j in range(Ny):
        x[index] = i * spacing
        y[index] = j * spacing
        index += 1

x /= max(x)
y /= max(y)
plt.scatter(x,y,s=points_radius**2)
def find_overlaps(x, y, r):
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(r)

    N = len(x)
    overlaps = []

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

### END ###

#initial overlaps - to make sure particles are separated
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
accepted_moves = 0

#initial MC simulation to determine initial acceptance ratio
for step in range(1000):
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
acceptance_ratio = accepted_moves/1000
print("initial acceptance ratio:", acceptance_ratio)

#equilibration
while acceptance_ratio>0.5 or acceptance_ratio<0.25:
    accepted_moves = 0
    if acceptance_ratio>0.5:
        d = d*1.05
        for j in range(0,1000):
            x, y, accepted = mc_move(x, y, r, d, L)
            if accepted:
                accepted_moves += 1
            if j % save_every_equil == 0:
                plt.figure()
                plt.scatter(x, y, s=points_radius**2)
                plt.xlim(0, L); plt.ylim(0, L)
                plt.title(f"Tuning Step {j}")
                plt.savefig(f"frames_equil/eq_{eq_frame_index:04d}.png")
                plt.close()
                eq_frame_index += 1
        acceptance_ratio = accepted_moves/1000
    if acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*0.95
        for j in range(0,1000):
            x, y, accepted = mc_move(x, y, r, d, L)
            if accepted:
                accepted_moves += 1
            if j % save_every_equil == 0:
                plt.figure()
                plt.scatter(x, y, s=points_radius**2)
                plt.xlim(0, L); plt.ylim(0, L)
                plt.title(f"Tuning Step {j}")
                plt.savefig(f"frames_equil/eq_{eq_frame_index:04d}.png")
                plt.close()
                eq_frame_index += 1
        acceptance_ratio = accepted_moves/1000
    print("Acceptance ratio:", acceptance_ratio)

print("optimal value of d: ", d)
plt.scatter(x,y,s=points_radius**2)
plt.show()
### MAIN SIMULATION + FRAME RECORDING ###
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

### BUILD EQUILIBRATION GIF ###
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

print("Equilibration GIF saved as equilibration.gif")

overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)