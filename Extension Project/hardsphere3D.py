#for putting all of the functions together
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt  # needed for 3D plotting
#eta is a random number
#delta is the amplitude of displacement
#moves are accepted/rejected when particles overlap/are separate
#this code is assuming no overlap between particles - purely for generating the moves


def fcc_lattice(N, r, L, tol=1e-8):
    if N % 4 != 0:
        raise ValueError("FCC lattice requires N to be multiple of 4")

    n_cells = int(round((N / 4)**(1/3)))
    if 4 * n_cells**3 != N:
        raise ValueError("N must equal 4 * n_cells^3")

    # Lattice constant
    a = L / n_cells

    # Nearest-neighbor spacing
    spacing = a / np.sqrt(2)

    if spacing < 2*r - tol:
        raise ValueError(
            f"FCC overlap unavoidable: spacing={spacing:.6f} < σ={2*r:.6f}"
        )

    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])

    coords = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    coords.append(a * (np.array([i, j, k]) + b))

    coords = np.array(coords)

    return coords[:,0], coords[:,1], coords[:,2], spacing



def mc_move(x, y, z, r, d, L):
    N = len(x)
    r = 0.03

    # Pick a random particle
    i = np.random.randint(N)
    old_x = x[i]
    old_y = y[i]
    old_z = z[i]

    # Propose move
    dx = d * (np.random.random_sample() - 0.5)
    dy = d * (np.random.random_sample() - 0.5)
    dz = d * (np.random.random_sample() - 0.5)
    # Apply move with periodic boundaries
    new_x = (x[i] + dx) % L
    new_y = (y[i] + dy) % L
    new_z = (z[i] + dz) % L
    # Temporarily update position for overlap check
    x[i] = new_x
    y[i] = new_y
    z[i] = new_z

    # Compute distances to all other particles with periodic boundaries
    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)

    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)

    dz_all = z[i] - z
    dz_all -= L * np.round(dz_all / L)

    dist2 = dx_all**2 + dy_all**2 + dz_all**2
    r_sum2 = 4*(r)**2
    dist2[i] = np.inf  # ignore self
    
    # Check for overlap
    if np.any(dist2 < r_sum2):
        # Reject move → restore old position
        x[i], y[i], z[i] = old_x, old_y, old_z
        return x, y, z, False

    return x, y, z, True

points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels
r = 0.03
a = np.pi*r**2
points_radius = 2 * r / 1.0 * points_whole_ax
# Setup
### GIF ADDITION ###
#os.makedirs("frames", exist_ok=True)
frame_index = 0
total_steps = 20000   # number of MC attempts
save_every = 100     # save a frame every N steps
### EQUILIBRATION GIF SETUP ###
#os.makedirs("frames_equil", exist_ok=True)
eq_frame_index = 0
save_every_equil = 100     # save a frame every 100 steps
d = 0.01
dr = 0.1*r
Nx = Ny = Nz = 3
N = 4*Nx*Ny*Nz
densities = np.array([0.68])
L_values = ((N*np.pi*(2*r)**3)/(6*(densities)))**(1/3)

#plt.scatter(x,y,s=points_radius**2)
def find_overlaps(x, y,z, r, L):  # Added L parameter
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    r = np.asarray(r)

    N = len(x)
    overlaps = []

    # Handle single radius case
    if r.ndim == 0:
        r = np.full(N, r)

    for i in range(N):
        for j in range(i+1, N): # Changed to avoid double counting   # Use periodic boundary conditions
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
                
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dz -= L * np.round(dz / L)
                
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < (r[i] + r[j]):
                overlaps.append((i, j))
    return overlaps

def sphere_function(xcentre, ycentre, zcentre,r):
    #draw sphere
    u,v = np.meshgrid(np.linspace(0,np.pi,51),np.linspace(0,2*np.pi, 101))
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    #print("x, y, z", x,y,z)
    #shift and scale sphere
    x = r*x + xcentre
    y = r*y + ycentre
    z = r*z + zcentre
    #print(x1,y1,z1)
    return x,y,z


def save_3d_frame_with_box(x, y, z, L, step, frames_dir, eq_frame_index, angle=None):
    """
    Save a 3D scatter plot of particles inside a cubic box for a GIF.

    Parameters
    ----------
    x, y, z : np.ndarray
        Particle positions
    L : float
        Box size
    step : int
        Current MC step (for title)
    frames_dir : str
        Directory to save frames
    eq_frame_index : int
        Frame number for filename
    angle : float or None
        Elevation/azimuth rotation for 3D view (optional)
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Particle scatter
    points_radius = 2 * r / 1.0 * points_whole_ax 

    for i in range (0,len(x)):
        xs,ys,zs = sphere_function(x[i],y[i],z[i],r) # same as before
        ax.plot_surface(xs, ys, zs, cmap = 'viridis')
    # Draw box edges
    # 8 corners of the cube
    corners = np.array([
        [0, 0, 0],
        [L, 0, 0],
        [L, L, 0],
        [0, L, 0],
        [0, 0, L],
        [L, 0, L],
        [L, L, L],
        [0, L, L]
    ])

    # List of edges as pairs of corner indices
    edges = [
        [0,1], [1,2], [2,3], [3,0], # bottom square
        [4,5], [5,6], [6,7], [7,4], # top square
        [0,4], [1,5], [2,6], [3,7]  # vertical edges
    ]

    for e in edges:
        ax.plot(
            [corners[e[0],0], corners[e[1],0]],
            [corners[e[0],1], corners[e[1],1]],
            [corners[e[0],2], corners[e[1],2]],
            color='black', linewidth=1
        )

    # Set axes limits and labels
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_box_aspect([1,1,1])  # keep cube proportions

    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    ax.set_title(f"η = {densities[density_idx]:.2f}, Step {step}", fontsize=20)

    # Rotate view for better 3D perception
    if angle is not None:
        ax.view_init(elev=30, azim=angle)

    plt.tight_layout()
    plt.savefig(f"{frames_dir}/eq_{eq_frame_index:04d}.png")
    plt.close()

def pairCorrelationFunction_3D(x, y, z, L, rMax, dr):
    rho = N/L**3

    nBins = int(rMax / dr)
    r_array = np.arange(dr/2, rMax, dr)  # bin centers
    g_r = np.zeros(len(r_array))  # will store g(r) values
    
    # Compute distances and bin them
    for i in range(N):
        for j in range(i+1, N):  # Only j > i to avoid double counting
            # Periodic boundary conditions
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dz -= L * np.round(dz / L)
            radius = np.sqrt(dx**2 + dy**2 + dz**2)
    
            if radius < rMax and radius>2*r:
                bin_idx = int(radius / dr) # bin number - what bin is the particle pair going to
                if bin_idx < len(g_r): #ensures no over-counting
                    g_r[bin_idx] += 2  # Count for both i-j and j-i - this is just the number of particles at this point
    
    # Normalize by ideal gas
    for i, r_val in enumerate(r_array):
        # Ideal number in shell: n_ideal = rho * 2*pi*r*dr
        n_ideal = rho * 4 * np.pi * (r_val)**2 * dr
        # Divide by N (total particles) and by n_ideal
        g_r[i] /= (N*n_ideal)
    
    return r_array, g_r

def averaged_g_r(x, y, z, r, d, L, rMax, dr, num_sims, sample):
    # Initialize variables for averaging
    g_r_sum = None
    sample_count = 0
    
    for i in range(num_sims):
        x, y, z, accepted = mc_move(x, y, z, r, d, L)
        
        if i % sample == 0:
            r_array, g_r_current = pairCorrelationFunction_3D(x, y, z, L, rMax, dr)
            print(i)
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
    
    # Create hexagonal lattice
    x, y, z, spacing = fcc_lattice(N, r, L)
    
    # Verify no overlaps and all particles in box
    overlaps = find_overlaps(x, y, z, r, L)
    print(f"L={L:.3f}: {len(overlaps)} overlaps, spacing={spacing:.4f}")
    print(f"All particles in box: {np.all((x >= 0) & (x < L) & (y >= 0) & (y < L))}")
    for step in range(10000000):
        if step % 100000 == 0:
            angle = (step // save_every) * 10  # rotate 10° per frame
            save_3d_frame_with_box(x, y, z, L, step, frames_dir, eq_frame_index, angle)
            eq_frame_index += 1
        x, y, z, accepted = mc_move(x, y, z, r, d, L)
        if accepted:
            accepted_moves += 1
        if step%100000 == 0 and step!=0:
            #print(step)
            #print(accepted_moves)
            acceptance_ratio = accepted_moves/step
            print("Acceptance ratio (no modification): ", acceptance_ratio)

    print("Acceptance ratio after 200,000 moves:", acceptance_ratio)
    while acceptance_ratio>0.5 or acceptance_ratio<0.25:
        if acceptance_ratio>0.5:
            accepted_moves=0
            d = d*1.05 
            print("d=", d)
            for j in range(0,10000):
                x,y,z,accepted = mc_move(x,y,z,r,d,L)
                if accepted:
                    accepted_moves+=1
            acceptance_ratio = accepted_moves/10000
            print("Acceptance ratio (large):", acceptance_ratio)
        if acceptance_ratio<0.25:
            accepted_moves = 0
            d = d*0.95
            print("d =",d)
            for j in range(0,10000):
                x,y,z,accepted = mc_move(x,y,z,r,d,L)
                if accepted:
                    accepted_moves+=1
            acceptance_ratio = accepted_moves/10000
            print("Acceptance ratio (small):", acceptance_ratio)
    print("final acceptance ratio:", acceptance_ratio)
    r_vals, g_r = averaged_g_r(x, y, z, r, d, L, 10*r, dr, 500000, 1000)
    
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
plt.savefig("g_r3D.png", dpi=300, bbox_inches='tight')
plt.show()

print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, z, r,L)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
#plt.scatter(x,y,s=points_radius**2)
#plt.show()