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

densities = np.array([0.68,0.71,0.74,0.77])
colors = ['red','blue','green','orange','purple','brown','black']

dr = 0.1 * r
rcut = 2**(1/6)*sigma

initial_d = 0.03

# ============================================================
#  HEXAGONAL LATTICE
# ============================================================

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
    
    return x, y

# ============================================================
#  LJ POTENTIAL (SHIFTED)
# ============================================================
def wca_potential(r, sigma, epsilon=1.0):
    rcut = 2**(1/6) * sigma
    
    if r >= rcut:
        return 0.0
    
    sr6 = (sigma / r)**6
    return 4 * epsilon * (sr6**2 - sr6) + epsilon
# ============================================================
#  METROPOLIS MC MOVE (LOCAL ΔU)
# ============================================================

def mc_move_lj(x, y, d, L, sigma, rcut, beta):
    i = np.random.randint(len(x))
    old_x, old_y = x[i], y[i]

    new_x = (old_x + d * (random.random() - 0.5)) % L
    new_y = (old_y + d * (random.random() - 0.5)) % L

    dU = 0.0

    for j in range(len(x)):
        if j == i:
            continue

        dx = old_x - x[j]
        dy = old_y - y[j]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        r_old = np.sqrt(dx*dx + dy*dy)

        dx = new_x - x[j]
        dy = new_y - y[j]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        r_new = np.sqrt(dx*dx + dy*dy)

        if r_old < rcut:
            dU -= wca_potential(r_old, sigma, rcut)
        if r_new < rcut:
            dU += wca_potential(r_new, sigma, rcut)

    if dU <= 0 or random.random() < np.exp(-beta * dU):
        x[i], y[i] = new_x, new_y
        return x,y,True
    return x,y,False

# ============================================================
#  RDF
# ============================================================

def pairCorrelationFunction_2D(x, y, L, rMax, dr):
    rho = len(x) / L**2
    r_vals = np.arange(dr/2, rMax, dr)
    g = np.zeros_like(r_vals)

    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            r = np.sqrt(dx*dx + dy*dy)
            if r < rMax:
                g[int(r/dr)] += 2

    for i, r in enumerate(r_vals):
        shell = 2 * np.pi * r * dr
        g[i] /= rho * len(x) * shell

    return r_vals, g

def averaged_g_r(x, y, r, d, L, rMax, dr, num_sims, sample, equil_steps=0, block_size=5):
    g_samples = []
    for i in range(num_sims):
        x, y, _ = mc_move_lj(x, y, d, L, sigma, rcut, beta)
        if i < equil_steps:
            continue
        if i % sample == 0:
            _, g_r_current = pairCorrelationFunction_2D(x, y, L, rMax, dr)
            print("g(r) sample: ", i)
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

# ============================================================
#  MAIN SIMULATION
# ============================================================

plt.figure(figsize=(10, 8))
rMax = 10*r
k=0
for idx, density in enumerate(densities):

    gr_file = f"gr_WCA_eta_{density:.2f}.npz"

    if os.path.exists(gr_file):
        print(f"Found existing data for η={density:.2f}. Loading instead of recomputing.")
        data = np.load(gr_file)
        r_vals = data["r_vals"]
        g_mean = data["g_mean"]
        g_err = data["g_err"]
    else:
        L = np.sqrt((N * np.pi * sigma**2) / (4 * density))
        x, y = create_hexagonal_lattice(N, r, L)

        d = initial_d
        accepted = 0
        attempted = 0
        frames_dir = f"frames_WCA_eta_{density:.2f}"
        os.makedirs(frames_dir, exist_ok=True)
        frame_idx = 0

        # ============================
        # EQUILIBRATION
        # ============================

        for step in range(0, 200_001):
            x,y,acc = mc_move_lj(x, y, d, L, sigma, rcut, beta)
            attempted += 1
            if acc:
                accepted += 1

            if step % 10_000 == 0 and step !=0:
                acc_ratio = accepted / attempted
                print(f"η={density:.2f} | step={step} | acc={acc_ratio:.3f} | d={d:.4f}")

                if acc_ratio > 0.5:
                    d *= 1.05
                elif acc_ratio < 0.25:
                    d *= 0.95

                accepted = 0
                attempted = 0
            if step % 10000 == 0:
                fig, ax = plt.subplots(figsize=(6,6))
                for j in range(N):
                    ax.add_patch(plt.Circle((x[j],y[j]), r, color='green', ec='black', alpha=0.8))
                ax.set_xlim(0,L)
                ax.set_ylim(0,L)
                ax.set_aspect('equal')
                ax.set_title(f'η={density:.2f}, step={step}', fontsize=18)
                ax.tick_params(axis='both', labelsize=16)
                plt.tight_layout()
                plt.savefig(f"{frames_dir}/frame_{frame_idx:04d}.png", dpi=150)
                plt.close(fig)
                frame_idx += 1
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        images = [Image.open(os.path.join(frames_dir,f)) for f in frame_files]
        if images:
            gif_path = f"equilibration_WCA_eta_{density:.2f}.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
            print(f"Saved GIF: {gif_path}")
        # ============================
        # RDF AVERAGING
        # ============================
        r_vals, g_mean, g_err = averaged_g_r(x,y,r,d,L,rMax,dr,500_000, 1000,equil_steps = 0, block_size=5)
        # =====================================
        # SAVE AVERAGED g(r) FOR THIS DENSITY
        # =====================================

        gr_file = f"gr_WCA_eta_{density:.2f}.npz"

        np.savez_compressed(
            gr_file,
            r_vals=r_vals,
            g_mean=g_mean,
            g_err=g_err,
            density=density,
            N=N,
            L=L,
            sigma=sigma,
            dr=dr,
            rMax=rMax
        )

        print(f"Saved averaged g(r) to {gr_file}")
        # Plot g(r) with error bands
    plt.errorbar(r_vals/(2*r), g_mean, yerr = g_err, linewidth=2, color=colors[idx], label=f'η={density:.2f}')
    #plt.fill_between(xplot, g_mean - g_err, g_mean + g_err, color=colors[k], alpha=0.3)

# ============================================================
#  FINAL PLOT
# ============================================================

plt.axhline(1.0, ls='--', color='black', alpha=0.5, label = 'Ideal gas')
plt.xlabel(r"$r/\sigma$", fontsize=18)
plt.ylabel(r"$g(r)$", fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.legend(ncols = 2, fontsize=18)
plt.tight_layout()
plt.savefig("wca2D_gr.png", dpi=300)
plt.show()