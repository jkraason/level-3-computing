import numpy as np
import matplotlib_inline as plt

import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
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
                bin_idx = int(radius / dr) # bin number
                if bin_idx < len(g_r): #ensures no over-counting
                    g_r[bin_idx] += 2  # Count for both i-j and j-i - this is just the number of particles at this point
    
    # Normalize by ideal gas
    for i, r_val in enumerate(r_array):
        # Ideal number in shell: n_ideal = rho * 2*pi*r*dr
        n_ideal = rho * 2 * np.pi * r_val * dr
        # Divide by N (total particles) and by n_ideal
        g_r[i] /= (N*n_ideal)
    
    return r_array, g_r

def averaged_g_r(Nx, Ny, spacing, r, d, L, rMax, dr, num_sims=10, equil_steps=10000):
    """Run multiple MC simulations and average g(r)"""
    g_r_sum = None
    
    for sim in range(num_sims):
        # Initialize
        x_sim = np.zeros(Nx*Ny)
        y_sim = np.zeros(Nx*Ny)
        index = 0
        for i in range(Nx):
            for j in range(Ny):
                x_sim[index] = i * spacing + np.random.uniform(-0.01, 0.01)
                y_sim[index] = j * spacing + np.random.uniform(-0.01, 0.01)
                index += 1
        x_sim /= max(x_sim)
        y_sim /= max(y_sim)
        
        # Equilibrate
        for step in range(equil_steps):
            x_sim, y_sim, _ = mc_move(x_sim, y_sim, r, d, L)
        
        # Calculate g(r)
        r_vals, g_r_single = pairCorrelationFunction_2D(x_sim, y_sim, L, rMax, dr)
        
        if g_r_sum is None:
            g_r_sum = g_r_single
        else:
            g_r_sum += g_r_single
    
    return r_vals, g_r_sum / num_sims


# Perform 10000 MC moves
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
accepted_moves = 0
for step in range(10000):
    x, y, accepted = mc_move(x, y, r, d, L)
    if accepted:
        accepted_moves += 1
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
        acceptance_ratio = accepted_moves/10000
        print("Acceptance ratio (large):", acceptance_ratio)
    if acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*0.95
        for j in range(0,10000):
            x,y,accepted = mc_move(x,y,r,d,L)
            if accepted:
                accepted_moves+=1
        acceptance_ratio = accepted_moves/10000
        print("Acceptance ratio (small):", acceptance_ratio)
print("final acceptance ratio:", acceptance_ratio)

N = len(x)
dr_values = [0.1*r,0.2*r,0.3*r,0.4*r,0.5*r]
g_r_results = []

print("\nCalculating g(r) for different dr values...")
for dr in dr_values:
    print(f"  Computing with dr = {dr:.4f}...")
    r_vals, g_r = averaged_g_r(Nx, Ny, spacing, r, d, L, L/2, dr, num_sims=10, equil_steps=10000)
    g_r_results.append((r_vals, g_r, dr))


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
points_whole_ax = 5 * 0.8 * 72
points_radius = 2 * r / 1.0 * points_whole_ax
plt.scatter(x, y, s=points_radius**2)
plt.xlim(0, L)
plt.ylim(0, L)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Configuration')
plt.gca().set_aspect('equal')

plt.subplot(1, 2, 2)
colors = ['blue', 'green', 'red', 'purple', 'orange']
for i, (r_vals, g_r, dr) in enumerate(g_r_results):
    plt.plot(r_vals, g_r, color=colors[i], linewidth=2, label=f'dr = {dr:.4f}')

plt.xlabel('r')
plt.ylabel('g(r)')
plt.title('Pair Correlation Function')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Ideal gas')
plt.legend()
plt.show()



print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, r)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
#plt.scatter(x,y,s=points_radius**2)
#plt.show()



