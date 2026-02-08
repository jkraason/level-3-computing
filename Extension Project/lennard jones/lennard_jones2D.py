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

densities = np.array([0.68])
colors = ['red', 'blue', 'green', 'purple']

dr = 0.1 * r
rcut = 10*r

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
            dU -= lj_shifted(r_old, sigma, rcut)
        if r_new < rcut:
            dU += lj_shifted(r_new, sigma, rcut)

    if dU <= 0 or random.random() < np.exp(-beta * dU):
        x[i], y[i] = new_x, new_y
        return True
    return False

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

# ============================================================
#  MAIN SIMULATION
# ============================================================

plt.figure(figsize=(10, 8))

for idx, density in enumerate(densities):

    L = np.sqrt((N * np.pi * sigma**2) / (4 * density))
    x, y = create_hexagonal_lattice(N, r, L)

    d = initial_d
    accepted = 0
    attempted = 0
    frames_dir = f"frames_LJ_eta_{density:.2f}"
    os.makedirs(frames_dir, exist_ok=True)
    frame_idx = 0

    # ============================
    # EQUILIBRATION
    # ============================

    for step in range(0, 200_001):
        acc = mc_move_lj(x, y, d, L, sigma, rcut, beta)
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
        if step%10000 == 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            for j in range(len(x)):
                ax.add_patch(plt.Circle((x[j], y[j]), r, color='blue', ec='black'))
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_aspect('equal')
            plt.savefig(f"{frames_dir}/frame_{frame_idx:04d}.png", dpi=150)
            plt.close()
            frame_idx += 1

    # ============================
    # RDF AVERAGING
    # ============================

    g_sum = None
    samples = 0

    for step in range(500_000):
        mc_move_lj(x, y, d, L, sigma, rcut, beta)
        if step % 1000 == 0:
            r_vals, g = pairCorrelationFunction_2D(x, y, L, L/2, dr)
            if g_sum is None:
                g_sum = np.zeros_like(g)
            g_sum += g
            samples += 1

    plt.plot(r_vals/sigma, g_sum/samples,
             lw=2, color=colors[idx],
             label=f"$\\eta={density:.2f}$")

# ============================================================
#  FINAL PLOT
# ============================================================

plt.axhline(1.0, ls='--', color='black', alpha=0.5)
plt.xlabel(r"$r/\sigma$", fontsize=18)
plt.ylabel(r"$g(r)$", fontsize=18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("lj_gr.png", dpi=300)
plt.show()
