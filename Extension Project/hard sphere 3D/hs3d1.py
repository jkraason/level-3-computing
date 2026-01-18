import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import glob
import os

# ----------------------
# 3D Lattice creation
# ----------------------
def create_hexagonal_lattice(N, r, L):
    """Create 3D Hexagonal Close-Packed (HCP) lattice"""
    a = 2.02 * r  # 1% buffer
    c = a * np.sqrt(8/3)  # Height between hexagonal layers
    
    # Estimate particles per layer
    n_per_layer = int(np.ceil(np.sqrt(N * 2/3)))
    
    x, y, z = [], [], []
    
    layer = 0
    particles_added = 0
    
    while particles_added < N:
        # ----- A layer (even layers) -----
        for i in range(n_per_layer):
            for j in range(n_per_layer):
                if particles_added >= N:
                    break
                
                # Hexagonal coordinates
                x_pos = i * a + (j % 2) * (a / 2)
                y_pos = j * a * np.sqrt(3) / 2
                
                # Apply periodic boundaries
                x_pos = x_pos % L
                y_pos = y_pos % L
                
                x.append(x_pos)
                y.append(y_pos)
                z.append(layer * c)  # A layer at base height
                particles_added += 1
                
                if particles_added >= N:
                    break
        
        if particles_added >= N:
            break
            
        # ----- B layer (odd layers) -----
        for i in range(n_per_layer):
            for j in range(n_per_layer):
                if particles_added >= N:
                    break
                
                # Offset for B layer
                x_pos = i * a + (j % 2) * (a / 2) + a/2
                y_pos = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                
                # Apply periodic boundaries
                x_pos = x_pos % L
                y_pos = y_pos % L
                
                x.append(x_pos)
                y.append(y_pos)
                z.append(layer * c + c/2)  # B layer at half height
                particles_added += 1
                
                if particles_added >= N:
                    break
        
        layer += 1
    
    # Take exactly N particles and convert to numpy arrays
    x = np.array(x[:N])
    y = np.array(y[:N])
    z = np.array(z[:N])
    
    # Center in box
    x = x - (x.max() + x.min())/2 + L/2
    y = y - (y.max() + y.min())/2 + L/2
    z = z - (z.max() + z.min())/2 + L/2
    
    # Apply periodic boundaries
    x = x % L
    y = y % L
    z = z % L
    
    return x, y, z, a

# ----------------------
# 3D Monte Carlo move
# ----------------------
def mc_move_3D(x, y, z, r, d, L):
    N = len(x)
    i = np.random.randint(N)
    old_x, old_y, old_z = x[i], y[i], z[i]

    dx = d * (random.random() - 0.5)
    dy = d * (random.random() - 0.5)
    dz = d * (random.random() - 0.5)

    x[i] = (x[i] + dx) % L
    y[i] = (y[i] + dy) % L
    z[i] = (z[i] + dz) % L

    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)
    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)
    dz_all = z[i] - z
    dz_all -= L * np.round(dz_all / L)

    dist2 = dx_all**2 + dy_all**2 + dz_all**2
    r_sum2 = (2*r)**2
    dist2[i] = np.inf

    if np.any(dist2 < r_sum2):
        x[i], y[i], z[i] = old_x, old_y, old_z
        return x, y, z, False

    return x, y, z, True

# ----------------------
# Helper function to add box wireframe
# ----------------------
def add_box_wireframe(ax, L):
    """Add wireframe box to 3D plot"""
    # Create wireframe box
    vertices = [
        [0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0],
        [0, 0, L], [L, 0, L], [L, L, L], [0, L, L]
    ]
    vertices = np.array(vertices)
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]
    
    for edge in edges:
        ax.plot([vertices[edge[0], 0], vertices[edge[1], 0]],
               [vertices[edge[0], 1], vertices[edge[1], 1]],
               [vertices[edge[0], 2], vertices[edge[1], 2]],
               'gray', alpha=0.5, linewidth=1)

# ----------------------
# Save 3D frame with nice formatting
# ----------------------
def save_3d_frame(x, y, z, L, r, eta, step, frame_dir, frame_index):
    """Save a nice 3D frame similar to 2D style"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate proper marker size (3D uses diameter in points, not area!)
    marker_size = 80 * (2*r) / L * 100
    
    # Color particles by z-coordinate for depth perception
    colors = plt.cm.viridis((z - z.min()) / (z.max() - z.min() + 1e-10))
    
    # Plot particles with nice styling
    scatter = ax.scatter(x, y, z, 
                        s=marker_size, 
                        c=colors, 
                        alpha=0.8,
                        edgecolors='black', 
                        linewidth=0.8,
                        depthshade=True)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    
    # Labels with nice formatting
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Z', fontsize=12, labelpad=10)
    
    # Title with simulation info
    title = f'3D Hard Spheres: η = {eta:.2f}, Step = {step}'
    if step == 0:
        title = f'Initial Configuration: η = {eta:.2f}'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set viewing angle (fixed for consistency)
    ax.view_init(elev=25, azim=45)
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Add box wireframe
    add_box_wireframe(ax, L)
    
    # Add colorbar for z-coordinate
    norm = plt.Normalize(z.min(), z.max())
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.1, label='Z coordinate')
    
    # Add text with simulation parameters
    stats_text = f'N = {len(x)}\nL = {L:.3f}\nr = {r:.3f}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    filename = os.path.join(frame_dir, f"frame_{frame_index:04d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

# ----------------------
# Other necessary functions
# ----------------------
def find_overlaps_3D(x, y, z, r, L):
    """Find all overlapping pairs with proper periodic boundary conditions"""
    N = len(x)
    overlaps = []
    
    # For checking all particles against each other
    for i in range(N):
        for j in range(i+1, N):
            # Compute minimum image distance with periodic boundaries
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            
            # Apply minimum image convention
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dz -= L * np.round(dz / L)
            
            # Calculate squared distance
            dist_sq = dx**2 + dy**2 + dz**2
            
            # Check if distance is less than 2r (or squared distance < (2r)^2)
            if dist_sq < (2*r)**2:
                overlaps.append((i, j, np.sqrt(dist_sq)))
    
    return overlaps
def pairCorrelationFunction_3D(x, y, z, L, rMax, dr):
    N = len(x)
    rho = N / L**3
    nBins = int(rMax / dr)
    r_array = np.arange(dr/2, rMax, dr)
    g_r = np.zeros(len(r_array))

    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dz -= L * np.round(dz / L)

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < rMax:
                bin_idx = int(dist / dr)
                g_r[bin_idx] += 2

    for i, r_val in enumerate(r_array):
        shell_volume = 4 * np.pi * r_val**2 * dr
        g_r[i] /= N * rho * shell_volume

    return r_array, g_r

def averaged_g_r_3D(x, y, z, r, d, L, rMax, dr, num_sims, sample):
    g_r_sum = None
    sample_count = 0
    for i in range(num_sims):
        x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
        if i % sample == 0:
            r_array, g_r_current = pairCorrelationFunction_3D(x, y, z, L, rMax, dr)
            if g_r_sum is None:
                g_r_sum = np.zeros_like(g_r_current)
            g_r_sum += g_r_current
            sample_count += 1

    if sample_count > 0:
        g_r_avg = g_r_sum / sample_count
    else:
        g_r_avg = np.array([])
    return r_array, g_r_avg

# ----------------------
# Main simulation
# ----------------------
# Parameters
Nx = Ny = 6  # Reduced for better visualization
Nz = 6
N = Nx * Ny * Nz 
r = 0.03
d = 0.02
dr = 0.1 * r
densities = np.array([0.68])
# CORRECTED 3D box size formula
L_values = ((N * np.pi * (2*r)**3) / (6 * densities))**(1/3)

# Create main output directory
os.makedirs("3D_simulation_frames", exist_ok=True)

# Run simulation
colors = ['red']
plt.figure(figsize=(10, 8))

for density_idx, (L, color) in enumerate(zip(L_values, colors)):
    eta = densities[density_idx]
    
    print(f"\n{'='*60}")
    print(f"Running 3D simulation for η = {eta:.2f}, L = {L:.3f}")
    print(f"{'='*60}")
    
    # Create frames directory for this density
    frames_dir = f"3D_simulation_frames/eta_{eta:.2f}"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize lattice
    x, y, z, spacing = create_hexagonal_lattice(N, r, L)
    overlaps = find_overlaps_3D(x, y, z, r, L)
    print(f"Initial configuration: {len(overlaps)} overlaps")
    print(f"Lattice spacing: {spacing:.4f}")
    print(f"Particle diameter: {2*r:.4f}")
    
    # Save initial frame
    save_3d_frame(x, y, z, L, r, eta, 0, frames_dir, 0)
    print("Saved initial configuration")
    
    # Equilibration
    accepted_moves = 0
    equilibration_steps = 20000
    
    for step in range(1, equilibration_steps + 1):
        x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
        if accepted:
            accepted_moves += 1
        
        # Save frames at intervals
        if step % 5000 == 0:
            frame_idx = step // 5000
            save_3d_frame(x, y, z, L, r, eta, step, frames_dir, frame_idx)
            print(f"Saved frame at step {step}")
        
        # Print progress
        if step % 5000 == 0:
            acceptance = accepted_moves / step
            print(f"  Step {step}: acceptance = {acceptance:.3f}")
    
    final_acceptance = accepted_moves / equilibration_steps
    print(f"Final acceptance after equilibration: {final_acceptance:.3f}")
    
    # Acceptance tuning (like your 2D code)
    print("\nTuning displacement for optimal acceptance...")
    while final_acceptance > 0.5 or final_acceptance < 0.25:
        if final_acceptance > 0.5:
            accepted_moves = 0
            d = d * 1.05
            print(f"  Increasing d to {d:.4f}")
        else:
            accepted_moves = 0
            d = d * 0.95
            print(f"  Decreasing d to {d:.4f}")
        
        # Test new d
        for j in range(5000):
            x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
            if accepted:
                accepted_moves += 1
        
        final_acceptance = accepted_moves / 5000
        print(f"  New acceptance: {final_acceptance:.3f}")
    
    print(f"Optimal d = {d:.4f}, acceptance = {final_acceptance:.3f}")
    
    # Save configuration after tuning
    save_3d_frame(x, y, z, L, r, eta, equilibration_steps, frames_dir, 4)
    
    # Production run for g(r)
    print("\nProduction run for g(r)...")
    production_steps = 10000
    accepted_moves = 0
    
    for step in range(production_steps):
        x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
        if accepted:
            accepted_moves += 1
    
    production_acceptance = accepted_moves / production_steps
    print(f"Production acceptance: {production_acceptance:.3f}")
    
    # Save final configuration
    save_3d_frame(x, y, z, L, r, eta, equilibration_steps + production_steps, 
                 frames_dir, 5)
    
    # Compute g(r)
    print("Computing g(r)...")
    rMax = 10 * r
    r_vals, g_r = averaged_g_r_3D(x, y, z, r, d, L, rMax, dr, 1000, 50)
    
    # Plot g(r)
    plt.plot(r_vals/(2*r), g_r, color=color, linewidth=2.5, 
            label=f'η = {eta:.2f}')
    
    # Create GIF from frames
    print("\nCreating GIF...")
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    frames = [Image.open(f) for f in frame_files]
    
    if len(frames) > 0:
        gif_path = f"3D_simulation_eta_{eta:.2f}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                     duration=500, loop=0)
        print(f"Saved GIF: {gif_path}")

# Final g(r) plot
plt.xlabel('r/σ', fontsize=14)
plt.ylabel('g(r)', fontsize=14)
plt.title('3D Hard Sphere Radial Distribution Function', fontsize=16)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Ideal gas')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig("3D_g_r_results.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SIMULATION COMPLETE")
print("="*60)
print(f"All frames saved in: 3D_simulation_frames/")
print(f"Final displacement: d = {d:.4f}")