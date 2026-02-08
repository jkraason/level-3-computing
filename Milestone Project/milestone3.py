import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

# ============================
# Create hexagonal lattice
# ============================
def create_hexagonal_lattice(N, r, L):
    spacing = 2.0 * r * 1.01  # 1% buffer to avoid overlaps
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    
    Nx = int(L / dx)
    Ny = int(L / dy)
    if Nx * Ny < N:
        spacing = 2.0 * r * 1.001
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        Nx = int(L / dx)
        Ny = int(L / dy)
    
    x = []
    y = []
    for i in range(Ny):
        for j in range(Nx):
            x_pos = j * dx + (i % 2) * (dx / 2)
            y_pos = i * dy
            x_pos = x_pos % L
            y_pos = y_pos % L
            if x_pos >= 0 and x_pos < L and y_pos >= 0 and y_pos < L:
                x.append(x_pos)
                y.append(y_pos)
            if len(x) >= N:
                break
        if len(x) >= N:
            break
    
    x = np.array(x[:N])
    y = np.array(y[:N])
    x_center = (x.max() + x.min()) / 2
    y_center = (y.max() + y.min()) / 2
    x = (x - x_center + L/2) % L
    y = (y - y_center + L/2) % L
    return x, y, spacing

# ============================
# Monte Carlo move
# ============================
def mc_move(x, y, r, d, L):
    N = len(x)
    i = np.random.randint(N)
    old_x, old_y = x[i], y[i]
    dx, dy = d * (random.random() - 0.5), d * (random.random() - 0.5)
    x[i] = (x[i] + dx) % L
    y[i] = (y[i] + dy) % L
    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)
    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)
    dist2 = dx_all**2 + dy_all**2
    dist2[i] = np.inf
    if np.any(dist2 < (2*r)**2):
        x[i], y[i] = old_x, old_y
        return x, y, False
    return x, y, True

# ============================
# Overlap check
# ============================
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
            dx = x[i]-x[j]; dx -= L*np.round(dx/L)
            dy = y[i]-y[j]; dy -= L*np.round(dy/L)
            if np.sqrt(dx**2+dy**2) < (r[i]+r[j]):
                overlaps.append((i,j))
    return overlaps

# ============================
# Pair correlation function
# ============================
def pairCorrelationFunction_2D(x, y, L, rMax, dr):
    N = len(x)
    rho = N / L**2
    nBins = int(rMax / dr)
    r_array = np.arange(dr/2, rMax, dr)
    g_r = np.zeros(len(r_array))
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i]-x[j]; dx -= L*np.round(dx/L)
            dy = y[i]-y[j]; dy -= L*np.round(dy/L)
            radius = np.sqrt(dx**2 + dy**2)
            if radius < rMax and radius > 2*r:
                bin_idx = int(radius / dr)
                if bin_idx < len(g_r):
                    g_r[bin_idx] += 2
    for i, r_val in enumerate(r_array):
        n_ideal = rho * 2 * np.pi * r_val * dr
        g_r[i] /= (N * n_ideal)
    return r_array, g_r

# ============================
# Averaged g(r) with error bands
# ============================
def averaged_g_r(x, y, r, d, L, rMax, dr, num_sims, sample, equil_steps=0, block_size=5):
    g_samples = []
    for i in range(num_sims):
        x, y, _ = mc_move(x, y, r, d, L)
        if i < equil_steps:
            continue
        if i % sample == 0:
            _, g_r_current = pairCorrelationFunction_2D(x, y, L, rMax, dr)
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

# ============================
# Simulation setup
# ============================
os.makedirs("frames", exist_ok=True)
os.makedirs("frames_equil", exist_ok=True)

r = 0.03
d = 0.02
dr = 0.1*r
Nx = Ny = 10
N = Nx * Ny
densities = np.array([0.72])
L_values = np.sqrt((N*np.pi*(2*r)**2)/(4*densities))
colors = ['red','blue','green','orange','purple']

# ============================
# Run simulation
# ============================
plt.figure(figsize=(10, 8))
k = 0
for density_idx, (L, color) in enumerate(zip(L_values, colors)):
    accepted_moves = 0
    eq_frame_index = 0
    frames_dir = f"frames_equil_eta_{densities[density_idx]:.2f}"
    os.makedirs(frames_dir, exist_ok=True)

    print(f"\n=== Running simulation for L = {L} ===")
    x, y, spacing = create_hexagonal_lattice(N, r, L)
    overlaps = find_overlaps(x, y, r, L)
    print(f"L={L:.3f}: {len(overlaps)} overlaps, spacing={spacing:.4f}")
    print(f"All particles in box: {np.all((x >= 0) & (x < L) & (y >= 0) & (y < L))}")

    # Run MC moves with original acceptance ratio logic
    for step in range(200001):
        if step % 10000 == 0:
            fig, ax = plt.subplots(figsize=(10,10))
            for j in range(len(x)):
                ax.add_patch(plt.Circle((x[j],y[j]), r, color='blue', ec='black', alpha=0.8, zorder=10, linewidth=2))
            ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_aspect('equal'); ax.grid(False)
            plt.xlabel('x', fontsize=20); plt.ylabel('y', fontsize=20)
            plt.title(f"η={densities[density_idx]:.2f}, Step {step}", fontsize=20)
            plt.savefig(f"{frames_dir}/eq_{eq_frame_index:04d}.png", dpi=150, bbox_inches='tight')
            plt.close()
            eq_frame_index += 1
        x, y, accepted = mc_move(x, y, r, d, L)
        if accepted:
            accepted_moves += 1
        if step % 10000 == 0 and step != 0:
            acceptance_ratio = accepted_moves / step
            print("Acceptance ratio:", acceptance_ratio)

    # Adjust d for acceptance ratio (kept original logic intact)
    while acceptance_ratio>0.5 or acceptance_ratio<0.25:
        if acceptance_ratio>0.5:
            accepted_moves=0
            d = d*1.05
            for j in range(10000):
                x,y,accepted = mc_move(x,y,r,d,L)
                if accepted:
                    accepted_moves+=1
            acceptance_ratio = accepted_moves/10000
        if acceptance_ratio<0.25:
            accepted_moves=0
            d = d*0.95
            for j in range(10000):
                x,y,accepted = mc_move(x,y,r,d,L)
                if accepted:
                    accepted_moves+=1
            acceptance_ratio = accepted_moves/10000

    # Compute g(r) with block-averaged error
    r_vals, g_r, g_err = averaged_g_r(x, y, r, d, L, rMax=10*r, dr=dr, num_sims=500_000, sample=500, equil_steps=50_000, block_size=5)

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
plt.savefig("2D g(r) error bands.png")
plt.show()

print("optimal value of d:", d)
overlaps = find_overlaps(x, y, r, L)
print("there are", len(overlaps), "overlaps")
print("Overlapping pairs:", overlaps)
