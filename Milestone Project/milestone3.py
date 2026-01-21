import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

def create_hexagonal_lattice(N, r, L):
    
    spacing = 2.0 * r * 1.01  # 1% buffer to ensure no overlaps
    
    # Hexagonal lattice geometry
    dx = spacing
    dy = spacing * np.sqrt(3) / 2  # vertical spacing for hex lattice
    
    # Calculate how many particles fit
    Nx = int(L / dx)
    Ny = int(L / dy)
    
    # Adjust if we need exactly N particles
    # (This will get approximately N particles)
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
    
    # Pick a random particle
    i = np.random.randint(N)
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

# Setup
os.makedirs("frames", exist_ok=True)
os.makedirs("frames_equil", exist_ok=True)

# Parameters
r = 0.03  # Particle radius
d = 0.02  # Initial displacement
dr = 0.1*r

Nx = 10
Ny = 10
N = Nx * Ny

densities = np.array([0.68, 0.69, 0.70, 0.71, 0.72])
L_values = np.sqrt((N*np.pi*(2*r)**2)/(4*(densities)))

def find_overlaps(x, y, r, L):
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.asarray(r)

    N = len(x)
    overlaps = []

    if r.ndim == 0:
        r = np.full(N, r)

    for i in range(N):
        for j in range(i+1, N):
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
    r_array = np.arange(dr/2, rMax, dr)
    g_r = np.zeros(len(r_array))
    
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            
            radius = np.sqrt(dx**2 + dy**2)
    
            if radius < rMax and radius>2*r:
                bin_idx = int(radius / dr)
                if bin_idx < len(g_r):
                    g_r[bin_idx] += 2
    
    for i, r_val in enumerate(r_array):
        n_ideal = rho * 2 * np.pi * r_val * dr
        g_r[i] /= (N*n_ideal)
    
    return r_array, g_r

def averaged_g_r(x, y, r, d, L, rMax, dr, num_sims, sample):
    g_r_sum = None
    sample_count = 0
    
    for i in range(num_sims):
        x, y, accepted = mc_move(x, y, r, d, L)
        
        if i % sample == 0:
            r_array, g_r_current = pairCorrelationFunction_2D(x, y, L, rMax, dr)
            
            if g_r_sum is None:
                g_r_sum = np.zeros_like(g_r_current)
            
            g_r_sum += g_r_current
            sample_count += 1
    
    if sample_count > 0:
        g_r_avg = g_r_sum / sample_count
    else:
        g_r_avg = np.array([])
    
    return r_array, g_r_avg

def calculate_marker_size(r, L, figsize=8, dpi=100):
    """
    Calculate proper marker size for circles with radius r in box of size L.
    
    The key is: marker size in scatter() is AREA in points^2.
    We want the visual diameter to be proportional to 2r/L.
    
    Parameters:
    r: radius of particles (data units)
    L: box size (data units)
    figsize: figure size in inches
    dpi: dots per inch
    
    Returns:
    marker_size: area in points^2 for scatter()
    """
    # Convert radius to diameter
    diameter = 2 * r
    
    # What fraction of the figure width should the diameter occupy?
    # For clear visibility, make particles occupy about 1/20th of figure width
    fraction_of_figure = diameter / L
    
    # Figure width in points (1 inch = 72 points)
    figure_width_points = figsize * 72
    
    # Desired diameter in points
    desired_diameter_points = fraction_of_figure * figure_width_points * 2.0  # Multiplier for visibility
    
    # Marker size is AREA in points^2 (π*(diameter/2)^2)
    # But matplotlib's scatter uses area, so we use diameter^2
    marker_size = (desired_diameter_points ** 2)
    
    return marker_size

# Alternative: Simple visual scaling
def get_visual_marker_size(r, L):
    """
    Simple visual scaling that works well for r=0.03
    """
    # Base size for good visibility
    base_size = 5000
    
    # Scale by (2r/L) to maintain relative size
    relative_size = (2 * r) / L
    
    # Additional scaling factor for visibility
    visibility_factor = 20  # Makes circles clearly visible
    
    marker_size = base_size * relative_size * visibility_factor
    
    return marker_size

# Even simpler: Hardcoded values that work
def get_good_marker_size_for_r_003(L):
    """
    Returns marker sizes that work well for r=0.03
    These are empirically determined
    """
    if L < 0.5:
        return 5000
    elif L < 1.0:
        return 3000
    elif L < 1.5:
        return 2000
    elif L < 2.0:
        return 1500
    else:
        return 1000

accepted_moves = 0
colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.figure(figsize=(10, 8))
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
    
    # Calculate marker size - using the simple function
    marker_size = get_good_marker_size_for_r_003(L)
    print(f"Box size L={L:.3f}, using marker size={marker_size:.0f}")
    
    for step in range(200001):
        if step % 10000 == 0:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.gca()
            
            # Plot with calculated marker size
            plt.scatter(x, y, s=marker_size, c='blue', edgecolors='black', 
                       linewidth=1.5, alpha=0.8, zorder=2)
            
            # Also plot actual circles for verification (optional)
            plot_circles = True
            if plot_circles and step == 0:  # Only for first frame
                for xi, yi in zip(x, y):
                    circle = plt.Circle((xi, yi), r, color='red', fill=False, 
                                       linestyle='--', linewidth=1, alpha=0.5, zorder=1)
                    ax.add_patch(circle)
            
            plt.xlabel('x', fontsize = 20)
            plt.ylabel('y', fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            plt.xlim(0, L)
            plt.ylim(0, L)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.title(f"η = {densities[density_idx]:.2f}, L = {L:.3f}, Step {step}\nParticle radius r = {r}", fontsize = 20)
            
            plt.savefig(f"{frames_dir}/eq_{eq_frame_index:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            eq_frame_index += 1
        
        x, y, accepted = mc_move(x, y, r, d, L)
        if accepted:
            accepted_moves += 1
        if step%10000 == 0 and step!=0:
            acceptance_ratio = accepted_moves/step
            print("Acceptance ratio (no modification): ", acceptance_ratio)

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
    plt.plot(r_vals/(2*r), g_r, color=color, linewidth=2, label=f'$\eta$ = '+str(format(densities[k], ".2f")))
    k=k+1

plt.xlabel('r/σ',fontsize = 20)
plt.ylabel('g(r)',fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal gas')
plt.legend(prop={'size': 15})
plt.savefig("multiple_L_g_r.png", dpi=300, bbox_inches='tight')
plt.show()

print("optimal value of d: ", d)
overlaps = find_overlaps(x, y, r, L)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)

# Create a demo plot with different marker sizes to find the right one
print("\n" + "="*60)
print("DEMONSTRATION OF DIFFERENT MARKER SIZES")
print("="*60)

# Create test configuration
test_L = 1.0
test_r = 0.03
test_N = 16

# Simple grid
x_test = np.linspace(test_r*2, test_L-test_r*2, 4)
y_test = np.linspace(test_r*2, test_L-test_r*2, 4)
X_test, Y_test = np.meshgrid(x_test, y_test)
x_test = X_test.ravel()
y_test = Y_test.ravel()

# Test different marker sizes
marker_sizes_to_test = [100, 500, 1000, 2000, 5000, 10000]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, marker_size in enumerate(marker_sizes_to_test):
    ax = axes[idx]
    
    # Plot with current marker size
    ax.scatter(x_test, y_test, s=marker_size, c='blue', 
              edgecolors='black', linewidth=1, alpha=0.8)
    
    # Add actual circles for comparison
    for xi, yi in zip(x_test, y_test):
        circle = plt.Circle((xi, yi), test_r, color='red', 
                          fill=False, linestyle='--', linewidth=1, alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_xlim(0, test_L)
    ax.set_ylim(0, test_L)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Marker size = {marker_size}\n(r={test_r}, L={test_L})', fontsize=10)
    
    # Check if scatter markers match circle outlines
    ax.text(0.05, 0.95, f'Size {marker_size}', transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))