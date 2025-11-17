#for putting all of the functions together
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
#eta is a random number
#delta is the amplitude of displacement
#moves are accepted/rejected when particles overlap/are separate
#this code is assuming no overlap between particles - purely for generating the moves

def mc_move(x, y, r, d, L):

    N = len(x)
    r=0.03


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
### GIF ADDITION ###
os.makedirs("frames", exist_ok=True)
frame_index = 0
total_steps = 20000   # number of MC attempts
save_every = 100     # save a frame every N steps
### EQUILIBRATION GIF SETUP ###
os.makedirs("frames_equil", exist_ok=True)
eq_frame_index = 0
save_every_equil = 100     # save a frame every 100 steps
d = 0.15
L = np.sqrt((100*np.pi*(0.06)**2)/(4*(0.68)))
print(L)

# Random initial positions
#file = 'Coordinates.dat'
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
N=len(x)
spacing = L/(Nx-1)   # > 2r prevents overlap

index = 0
for i in range(Nx):
    for j in range(Ny):
        x[index] = (i) * spacing
        y[index] = (j) * spacing
        index += 1
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
dr=0.1*r
N=len(x)
def pairCorrelationFunction_2D(x, y, L, rMax, dr):
    rho = N/L**2
    #r=0.3
    nBins = int(rMax / dr)
    r_array = np.arange(dr/2, rMax, dr)  # bin centers
    g_r = np.zeros(len(r_array))  # will store g(r) values
    
    # Compute distances and bin them
    for i in range(N):
        for j in range(i+1, N):  # Only j > i to avoid double counting
            # Periodic boundary conditions
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            
            radius = np.sqrt(dx**2 + dy**2)
    
            if radius < rMax and radius>2*r:
                bin_idx = int(radius / dr) # bin number - what bin is the particle pair going to
                if bin_idx < len(g_r): #ensures no over-counting
                    g_r[bin_idx] += 2  # Count for both i-j and j-i - this is just the number of particles at this point
    
    # Normalize by ideal gas
    for i, r_val in enumerate(r_array):
        # Ideal number in shell: n_ideal = rho * 2*pi*r*dr
        n_ideal = rho * 2 * np.pi * r_val * dr
        # Divide by N (total particles) and by n_ideal
        g_r[i] /= (N*n_ideal)
    
    return r_array, g_r

def averaged_g_r(x, y, r, d, L, rMax, dr, num_sims, equil_steps):

    g_r_sum = None

    # Make copies so the original arrays are not modified
    x_sim = np.copy(x)
    y_sim = np.copy(y)

    for sim in range(num_sims):
        # Equilibrate: perform MC moves
        for step in range(equil_steps):
            x_sim, y_sim, _ = mc_move(x_sim, y_sim, r, d, L)

        # Compute g(r) for this snapshot
        r_vals, g_r_single = pairCorrelationFunction_2D(x_sim, y_sim, L, rMax, dr)

        # Accumulate
        if g_r_sum is None:
            g_r_sum = g_r_single
        else:
            g_r_sum += g_r_single

    # Average over snapshots
    g_r_avg = g_r_sum / num_sims

    return r_vals, g_r_avg


# Perform 1000 MC moves
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
accepted_moves = 0
for step in range(200001):
    if step % 10000 == 0:
        plt.figure()
        plt.scatter(x, y, s=points_radius**2)
        plt.xlabel('x', fontsize = 15)
        plt.ylabel('y', fontsize = 15)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xlim(0, L); plt.ylim(0, L)
        plt.title(f"Equilibration Step {step}", fontsize = 15)
        plt.savefig(f"frames_equil/eq_{eq_frame_index:04d}.png")
        plt.close()
        eq_frame_index += 1
    x, y, accepted = mc_move(x, y, r, d, L)
    if accepted:
        accepted_moves += 1
    if step%10000 == 0 and step!=0:
        #print(step)
        #print(accepted_moves)
        acceptance_ratio = accepted_moves/step
        print("Acceptance ratio (no modification): ", acceptance_ratio)

        
#run simulation
print("Acceptance ratio after 200,000 moves:", acceptance_ratio)
while acceptance_ratio>0.5 or acceptance_ratio<0.25:
    if acceptance_ratio>0.5:
        accepted_moves=0
        d = d*1.05 
        print("d=", d)
        for j in range(0,10000):
            x,y,accepted = mc_move(x,y,r,d,L)
            if accepted:
                accepted_moves+=1
        acceptance_ratio = accepted_moves/10000
        print("Acceptance ratio (large):", acceptance_ratio)
    if acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*0.95
        print("d =",d)
        for j in range(0,10000):
            x,y,accepted = mc_move(x,y,r,d,L)
            if accepted:
                accepted_moves+=1
        acceptance_ratio = accepted_moves/10000
        print("Acceptance ratio (small):", acceptance_ratio)
print("final acceptance ratio:", acceptance_ratio)



N = len(x)
dr_values = [0.1*r,0.2*r,0.3*r,0.4*r,0.5*r]
dr_text = ['δr = 0.1r', 'δr = 0.2r', 'δr = 0.3r', 'δr = 0.4r', 'δr = 0.5r']
g_r_results = []

print("\nCalculating g(r) for different dr values...")
for dr in dr_values:
    print(f"  Computing with dr = {dr:.4f}...")
    r_vals, g_r = averaged_g_r(x,y,r,d,L,L/2,dr,num_sims=100, equil_steps=1000)
    g_r_results.append((r_vals, g_r, dr))

plt.figure(figsize=(5,5))
fig = plt.gcf()
dpi = fig.get_dpi()
colors = ['grey', 'green', 'red', 'purple', 'orange', 'yellow']
for i, (r_vals, g_r, dr) in enumerate(g_r_results):#enumerate removes the need for a counter variable - improves efficiency
    plt.plot(r_vals, g_r, color=colors[i], linewidth=1, label = dr_text[i])

plt.xlabel('r')
plt.ylabel('g(r)')
plt.title('Pair Correlation Function')
plt.grid(False)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Ideal gas')
plt.savefig("pair_correlation.png", dpi=300, bbox_inches='tight')
plt.legend()
plt.show()
print("there are", N, "particles")
### BUILD EQUILIBRATION GIF ###
eq_frames = []
files = sorted(glob.glob("frames_equil/*.png"))

for f in files:
    eq_frames.append(Image.open(f))

eq_frames[0].save(
    "milestone.gif",
    save_all=True,
    append_images=eq_frames[1:],
    duration=200,
    loop=0
)

print("Equilibration GIF saved as equilibration.gif")
print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
#plt.scatter(x,y,s=points_radius**2)
#plt.show()