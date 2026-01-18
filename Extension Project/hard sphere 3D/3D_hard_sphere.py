import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

# ----------------------
# 3D Lattice creation
# ----------------------
def create_fcc_lattice(N, r, L):
    # FCC lattice spacing
    a = 2 * r * np.sqrt(2) * 1.01  # 1% buffer
    
    # Number of unit cells needed
    n_cells = int(np.ceil((N / 4)**(1/3)))
    
    x, y, z = [], [], []
    
    # FCC basis vectors
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    # Generate positions
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                origin = np.array([i, j, k]) * a
                for b in basis:
                    if len(x) >= N:
                        break
                    pos = origin + b
                    
                    # Apply periodic boundaries immediately
                    x_pos = pos[0] % L
                    y_pos = pos[1] % L
                    z_pos = pos[2] % L
                    
                    x.append(x_pos)
                    y.append(y_pos)
                    z.append(z_pos)
    
    x = np.array(x[:N])
    y = np.array(y[:N])
    z = np.array(z[:N])
    
    # NO CENTERING NEEDED - particles are already in [0, L)
    # The issue was the centering step pushing particles outside
    
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
# 3D overlap detection
# ----------------------
def find_overlaps_3D(x, y, z, r, L):
    N = len(x)
    overlaps = []
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]

            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            dz -= L * np.round(dz / L)

            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist < 2*r:
                overlaps.append((i, j))
    return overlaps

# ----------------------
# 3D pair correlation
# ----------------------
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

# ----------------------
# Average g(r) over MC moves
# ----------------------
def averaged_g_r_3D(x, y, z, r, d, L, rMax, dr, num_sims, sample):
    g_r_sum = None
    sample_count = 0
    for i in range(num_sims):
        x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
        if i % sample == 0:
            print()
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
# Simulation parameters
# ----------------------
Nx = Ny = 10
Nz = 10
N = Nx * Ny * Nz
r = 0.03
d = 0.02
dr = 0.1*r
densities = np.array([0.68])
L_values = np.sqrt((N*np.pi*(2*r)**2)/(4*densities))  # approximate 3D L

# Frame directories
os.makedirs("frames_3D", exist_ok=True)

# ----------------------
# Run simulation for each density
# ----------------------
colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.figure(figsize=(8,6))
k = 0

for density_idx, (L, color) in enumerate(zip(L_values, colors)):
    accepted_moves = 0
    eq_frame_index = 0
    frames_dir = f"frames_3D_eta_{densities[density_idx]:.2f}"
    os.makedirs(frames_dir, exist_ok=True)

    print(f"\n=== Running simulation for L = {L} ===")

    # Initialize 3D lattice
    x, y, z, spacing = create_fcc_lattice(N, r, L)
    overlaps = find_overlaps_3D(x, y, z, r, L)
    print(f"L={L:.3f}: {len(overlaps)} overlaps, spacing={spacing:.4f}")

    # Equilibration and frame saving
    for step in range(20001):
        if step % 5000 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, s=20)
            ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
            ax.view_init(elev=30, azim=step/20000*360)  # rotate for visualization
            plt.savefig(f"{frames_dir}/eq_{eq_frame_index:04d}.png")
            plt.close()
            eq_frame_index += 1

        x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
        if accepted:
            accepted_moves += 1

    acceptance_ratio = accepted_moves / 20001
    print("Acceptance ratio after equilibration:", acceptance_ratio)

    # Adjust d for reasonable acceptance (0.25-0.5)
    while acceptance_ratio>0.5 or acceptance_ratio<0.25:
        accepted_moves = 0
        d = d*1.05 if acceptance_ratio>0.5 else d*0.95
        for _ in range(10000):
            x, y, z, accepted = mc_move_3D(x, y, z, r, d, L)
            if accepted: accepted_moves += 1
        acceptance_ratio = accepted_moves / 10000
        print(f"Adjusted d={d:.5f}, acceptance={acceptance_ratio:.3f}")

    # Compute g(r)
    r_vals, g_r = averaged_g_r_3D(x, y, z, r, d, L, 10.5*r, dr, 500, 10)

    # Build GIF
    eq_frames = []
    files = sorted(glob.glob(f"{frames_dir}/*.png"))
    for f in files:
        eq_frames.append(Image.open(f))

    if len(eq_frames) > 0:
        eq_frames[0].save(
            f"equilibration_3D_eta_{densities[density_idx]:.2f}.gif",
            save_all=True,
            append_images=eq_frames[1:],
            duration=200,
            loop=0
        )
        print(f"Saved GIF: equilibration_3D_eta_{densities[density_idx]:.2f}.gif")

    # Plot g(r)
    plt.plot(r_vals/(2*r), g_r, color=color, linewidth=2, label=f'$\eta$ = {densities[k]:.2f}')
    k += 1

plt.xlabel('r/σ', fontsize=15)
plt.ylabel('g(r)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Ideal gas')
plt.legend(prop={'size': 15})
plt.savefig("multiple_L_g_r_3D.png", dpi=300, bbox_inches='tight')
plt.show()

print("Optimal value of d:", d)
overlaps = find_overlaps_3D(x, y, z, r, L)
print("Number of overlaps after simulation:", len(overlaps))
