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

def create_hexagonal_lattice(N, r, L):
    
    # Hexagonal lattice spacing
    # For hard disks, minimum spacing is 2*r (touching)
    spacing = 2.01 * r  # 1% buffer to ensure no overlaps
    
    # Hexagonal lattice geometry
    dx = spacing
    dy = spacing 
    
    # Calculate how many particles fit
    Nx = int(L / dx)
    Ny = int(L / dy)

    if Nx * Ny < N:
        # Need tighter packing
        spacing = 2.0 * r * 1.001
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        Nx = int(L / dx)
        Ny = int(L / dy)
    
    x = []
    y = []
    
    for i in range(Ny):
        for j in range(Nx):
            # x position: shift every other row by dx/2 for hexagonal packing
            x_pos = j * dx + (i % 2) * (dx / 2)
            y_pos = i * dy
            
            # Apply periodic boundary conditions
            x_pos = x_pos % L
            y_pos = y_pos % L
            
            # Only add if within bounds (with margin for particle radius)
            if x_pos >= 0 and x_pos < L and y_pos >= 0 and y_pos < L:
                x.append(x_pos)
                y.append(y_pos)
            
            if len(x) >= N:
                break
        if len(x) >= N:
            break
    
    # Center the lattice in the box
    x = np.array(x[:N])
    y = np.array(y[:N])
    
    # Center the configuration
    x_center = (x.max() + x.min()) / 2
    y_center = (y.max() + y.min()) / 2
    
    x = x - x_center + L / 2
    y = y - y_center + L / 2
    
    # Apply periodic boundaries
    x = x % L
    y = y % L
    
    return x, y, spacing

def mc_move(x, y, r, d, L):
    N = len(x)
    r = 0.03

    # Pick a random particle
    i = np.random.randint(N)
    old_x = x[i]
    old_y = y[i]

    # Save old position
    old_x = x[i]
    old_y = y[i]

    # Propose move
    dx = d * (random.random() - 0.5)
    dy = d * (random.random() - 0.5)
    
    # Apply move with periodic boundaries
    new_x = (x[i] + dx) % L
    new_y = (y[i] + dy) % L

    # Temporarily update position for overlap check
    x[i] = new_x
    y[i] = new_y

    # Compute distances to all other particles with periodic boundaries
    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)

    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)

    dist2 = dx_all**2 + dy_all**2
    r_sum2 = 4*(r)**2
    dist2[i] = np.inf  # ignore self
    
    # Check for overlap
    if np.any(dist2 < r_sum2):
        # Reject move → restore old position
        x[i], y[i] = old_x, old_y
        return x, y, False

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
d = 0.02
dr = 0.1*r

Nx = 10
Ny = 10
N = Nx * Ny

densities = np.array([0.68,0.70,0.72])
L_values = np.sqrt((N*np.pi*(2*r)**2)/(4*(densities)))

#plt.scatter(x,y,s=points_radius**2)
def find_overlaps(x, y, r, L):  # Added L parameter
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(r)

    N = len(x)
    overlaps = []

    # Handle single radius case
    if r.ndim == 0:
        r = np.full(N, r)

    for i in range(N):
        for j in range(i+1, N):  # Changed to avoid double counting
            # Use periodic boundary conditions
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist < (r[i] + r[j]):
                overlaps.append((i, j))
    return overlaps

def pairCorrelationFunction_2D(x, y, L, rMax, dr):
    rho = N/L**2

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

def averaged_g_r(x, y, r, d, L, rMax, dr, num_sims, sample):
    # Initialize variables for averaging
    g_r_sum = None
    sample_count = 0
    
    for i in range(num_sims):
        x, y, accepted = mc_move(x, y, r, d, L)
        
        if i % sample == 0:
            r_array, g_r_current = pairCorrelationFunction_2D(x, y, L, rMax, dr)
            
            # Initialize g_r_sum on first sample
            if g_r_sum is None:
                g_r_sum = np.zeros_like(g_r_current)
            
            # Accumulate the current g(r)
            g_r_sum += g_r_current
            sample_count += 1
    
    # Calculate the average
    if sample_count > 0:
        g_r_avg = g_r_sum / sample_count
    else:
        g_r_avg = np.array([])  # or handle the no-samples case appropriately
    
    return r_array, g_r_avg

accepted_moves = 0
colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.figure(figsize=(8,6))
k=0
for density_idx, (L, color) in enumerate(zip(L_values, colors)):
    
    accepted_moves = 0
    eq_frame_index = 0
    
    # Create separate folder for this density
    frames_dir = f"frames_equil_eta_{densities[density_idx]:.2f}"
    os.makedirs(frames_dir, exist_ok=True)
    
    #lattice - initialise for each L
    print(f"\n=== Running simulation for L = {L} ===")
    r = 0.03
    d = 0.02
    
    # Create hexagonal lattice
    x, y, spacing = create_hexagonal_lattice(N, r, L)
    
    # Verify no overlaps and all particles in box
    overlaps = find_overlaps(x, y, r, L)
    print(f"L={L:.3f}: {len(overlaps)} overlaps, spacing={spacing:.4f}")
    print(f"All particles in box: {np.all((x >= 0) & (x < L) & (y >= 0) & (y < L))}")

    overlaps = find_overlaps(x, y, r, L)
    print("there are", len(overlaps), "overlaps")
    for step in range(200001):
        if step % 10000 == 0:
            plt.figure()
            plt.scatter(x, y, s=points_radius**2)
            plt.xlabel('x', fontsize = 15)
            plt.ylabel('y', fontsize = 15)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.xlim(0, L); plt.ylim(0, L)
            plt.title(f"η = {densities[density_idx]:.2f}, Step {step}", fontsize = 15)
            plt.savefig(f"{frames_dir}/eq_{eq_frame_index:04d}.png")
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
    r_vals, g_r = averaged_g_r(x, y, r, d, L, 10*r, dr, 500000, 1000)
    
    # Build GIF for this density
    eq_frames = []
    files = sorted(glob.glob(f"{frames_dir}/*.png"))
    for f in files:
        eq_frames.append(Image.open(f))
    
    if len(eq_frames) > 0:
        eq_frames[0].save(
            f"equilibration_eta_{densities[density_idx]:.2f}.gif",
            save_all=True,
            append_images=eq_frames[1:],
            duration=200,
            loop=0
        )
        print(f"Saved GIF: equilibration_eta_{densities[density_idx]:.2f}.gif")
    
    # Plot for this L value
    plt.plot(r_vals/(2*r), g_r, color=color, linewidth=2, label=f'$\eta$ ='+ str(format(densities[k],".2f")))
    k=k+1
plt.xlabel('r/σ')
plt.ylabel('g(r)')
plt.title('Pair Correlation Function for Different Box Sizes')
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal gas')
plt.legend()
plt.savefig("multiple_L_g_r.png", dpi=300, bbox_inches='tight')
plt.show()

print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, r,L)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
#plt.scatter(x,y,s=points_radius**2)
#plt.show()