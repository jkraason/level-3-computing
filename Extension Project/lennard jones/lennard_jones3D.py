import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
#  SYSTEM PARAMETERS

N=100

r = 0.03
sigma = 2 * r
epsilon = 1.0

kB = 1.0
T = 1.0
beta = 1.0 / (kB * T)

densities = np.array([0.46,0.50,0.54,0.58,0.62,0.66])
L_values = ((N*np.pi*(2*r)**3)/(6*(densities)))**(1/3)
colors = ['red', 'blue', 'green', 'purple']

dr = 0.1 * r
rcut = 10*r

initial_d = 0.03
points_whole_ax = 5 * 0.8 * 72 

save_every = 100

# ============================================================
#  HEXAGONAL LATTICE
# ============================================================

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

# ============================================================
#  LJ POTENTIAL (SHIFTED)
# ============================================================

def lj_potential(r, sigma, epsilon=1.0):
    sr6 = (sigma / r) ** 6
    return 4 * epsilon * (sr6**2 - sr6)

def lj_shifted(r, sigma, rcut):
    if r >= rcut:
        return 0.0
    return lj_potential(r, sigma) - lj_potential(rcut, sigma)

# ============================================================
#  METROPOLIS MC MOVE (LOCAL ΔU)
# ============================================================

def mc_move_lj(x, y, z, d, L, sigma, rcut, beta):
    i = np.random.randint(len(x))
    old_x, old_y, old_z = x[i], y[i], z[i]

    new_x = (old_x + d * (random.random() - 0.5)) % L
    new_y = (old_y + d * (random.random() - 0.5)) % L
    new_z = (old_z + d * (random.random() - 0.5)) % L

    dU = 0.0

    for j in range(len(x)):
        if j == i:
            continue

        dx = old_x - x[j]
        dy = old_y - y[j]
        dz = old_z - z[j]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        dz -= L * np.round(dz / L)
        r_old = np.sqrt(dx*dx + dy*dy + dz*dz)

        dx = new_x - x[j]
        dy = new_y - y[j]
        dz = new_z - z[j]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        dz -= L * np.round(dz / L)
        r_new = np.sqrt(dx*dx + dy*dy + dz*dz)

        if r_old < rcut:
            dU -= lj_shifted(r_old, sigma, rcut)
        if r_new < rcut:
            dU += lj_shifted(r_new, sigma, rcut)

    if dU <= 0 or random.random() < np.exp(-beta * dU):
        x[i], y[i], z[i] = new_x, new_y, new_z
        return x,y,z,True
    return x,y,z,False

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
    ax.grid(False)

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

# ============================================================
#  RDF
# ============================================================

def pairCorrelationFunction_3D(x, y, z, L, rMax, dr):
    rho = len(x) / L**2
    r_vals = np.arange(dr/2, rMax, dr)
    g = np.zeros_like(r_vals)

    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dz -= L * np.round(dz / L)
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            if r < rMax:
                g[int(r/dr)] += 2

    for i, r in enumerate(r_vals):
        shell = 2 * np.pi * r * dr
        g[i] /= rho * len(x) * shell

    return r_vals, g

# ============================================================
#  MAIN SIMULATION
# ============================================================

def averaged_g_r(x, y,z, r, d, L, rMax, dr, num_sims, sample, equil_steps=0, block_size=5):
    g_samples = []
    for i in range(num_sims):
        x, y, z, _ = mc_move_lj(x, y, z, r, d, L)
        if i < equil_steps:
            continue
        if i % sample == 0:
            _, g_r_current = pairCorrelationFunction_3D(x, y, z, L, rMax, dr)
            g_samples.append(g_r_current)
    g_samples = np.array(g_samples)
    n_samples, n_bins = g_samples.shape
    n_blocks = n_samples // block_size
    trimmed = g_samples[:n_blocks*block_size]
    blocks = trimmed.reshape(n_blocks, block_size, n_bins)
    block_means = blocks.mean(axis=1)
    g_mean = block_means.mean(axis=0)
    g_var = np.var(block_means, axis=0, ddof=1)
    g_err = np.sqrt(g_var / n_blocks)
    r_array = np.arange(dr/2, rMax, dr)
    return r_array, g_mean, g_err


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
    
    # Create fcc lattice
    x, y, z, spacing = fcc_lattice(N, r, L)
    
    for step in range(10000000):
        if step % 100000 == 0:
            angle = (400000 // save_every) * 10  # rotate 10° per frame
            save_3d_frame_with_box(x, y, z, L, step, frames_dir, eq_frame_index, angle)
            eq_frame_index += 1
        x, y, z, accepted = mc_move_lj(x, y, z, r, d, L)
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
                x,y,z,accepted = mc_move_lj(x,y,z,r,d,L)
                if accepted:
                    accepted_moves+=1
            acceptance_ratio = accepted_moves/10000
            print("Acceptance ratio (large):", acceptance_ratio)
        if acceptance_ratio<0.25:
            accepted_moves = 0
            d = d*0.95
            print("d =",d)
            for j in range(0,10000):
                x,y,z,accepted = mc_move_lj(x,y,z,r,d,L)
                if accepted:
                    accepted_moves+=1
            acceptance_ratio = accepted_moves/10000
            print("Acceptance ratio (small):", acceptance_ratio)
    print("final acceptance ratio:", acceptance_ratio)
    r_vals, g_r = averaged_g_r(x, y, z, r, d, L, L/2, dr, 500000, 1000)
    
    # Build GIF for this density
    eq_frames = []
    files = sorted(glob.glob(f"{frames_dir}/*.png"))
    for f in files:
        eq_frames.append(Image.open(f))
    
    if len(eq_frames) > 0:
        eq_frames[0].save(
            f"3D_equilibration_eta_{densities[density_idx]:.2f}.gif",
            save_all=True,
            append_images=eq_frames[1:],
            duration=200,
            loop=0
        )
        print(f"Saved GIF: 3D_equilibration_eta_{densities[density_idx]:.2f}.gif")
    
    # Plot for this L value
    r_vals, g_r, g_err = averaged_g_r(x, y, r, d, L, rMax=L/2, dr=dr, num_sims=500_000, sample=500, equil_steps=50_000, block_size=5)

    # Plot g(r) with error bands
    xplot = r_vals / (2*r)
    plt.plot(xplot, g_r, color=color, linewidth=2, label=fr'$\eta={densities[k]:.2f}$')
    plt.fill_between(xplot, g_r - g_err, g_r + g_err, color=color, alpha=0.3)
    k += 1

plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel(r"$r/\sigma$")
plt.ylabel(r"$g(r)$")
plt.legend()
plt.tight_layout()
plt.savefig("3D g(r) error bands.png")
plt.show()