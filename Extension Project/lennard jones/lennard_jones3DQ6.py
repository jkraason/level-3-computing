import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
import glob
#  SYSTEM PARAMETERS
Nx=Ny=Nz=3
N=4*Nx*Ny*Nz

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

# ================================
# Neighbor list
# ================================
def find_neighbors(x, y, z, L, rcut):
    N = len(x)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dz = z[i]-z[j]

            dx -= L*np.round(dx/L)
            dy -= L*np.round(dy/L)
            dz -= L*np.round(dz/L)

            if np.sqrt(dx**2 + dy**2 + dz**2) < rcut:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors

# ================================
# Q6 calculation
# ================================
def compute_Q6(x, y, z, neighbors, L):
    N = len(x)
    l = 6
    qlm = np.zeros((N, 2*l+1), dtype=complex)

    for i in range(N):
        if len(neighbors[i]) == 0:
            continue
        for j in neighbors[i]:
            dx = x[j]-x[i]
            dy = y[j]-y[i]
            dz = z[j]-z[i]

            dx -= L*np.round(dx/L)
            dy -= L*np.round(dy/L)
            dz -= L*np.round(dz/L)

            r = np.sqrt(dx**2 + dy**2 + dz**2)
            if r == 0: continue
            theta = np.arccos(dz/r)
            phi = np.arctan2(dy, dx)

            for m in range(-l, l+1):
                qlm[i,m+l] += sph_harm(m, l, phi, theta)

        qlm[i] /= len(neighbors[i])

    Q6_local = np.sqrt((4*np.pi/(2*l+1)) * np.sum(np.abs(qlm)**2, axis=1))
    Q6_global = np.mean(Q6_local)
    return Q6_local, Q6_global

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

def block_average_error(data, block_size):
    data = np.asarray(data)
    N = len(data)
    n_blocks = N // block_size
    if n_blocks < 2:
        return np.nan, np.nan
    blocks = data[:n_blocks*block_size].reshape(n_blocks, block_size)
    block_means = blocks.mean(axis=1)
    mean = block_means.mean()
    variance = np.sum((block_means - mean)**2)/(n_blocks-1)
    error = np.sqrt(variance / n_blocks)
    return mean, error

r = 0.03
d = 0.001
dr = 0.1*r
rcut = 1.4*(2*r)
Nx = Ny = Nz = 3
N = 4*Nx*Ny*Nz

densities = np.array([0.49,0.51,0.53,0.55])
colors = ['red','blue','green','purple']
L_values = ((N*np.pi*(2*r)**3)/(6*densities))**(1/3)

# ================================
# Storage for Q6
# ================================
Q6_all = {}
step_all = {}
Q6_means = []
Q6_errors = []

# ================================
# MAIN LOOP
# ================================
for density_idx, (L, color) in enumerate(zip(L_values, colors)):
    print(f"\n=== Density η = {densities[density_idx]:.2f} ===")

    Q6_history = []
    step_history = []

    x, y, z, spacing = fcc_lattice(N, r, L)
    accepted_moves = 0

    for step in range(200000):
        x, y, z, acc = mc_move_lj(x, y, z, r, d, L, rcut, beta)
        accepted_moves += acc

        # Compute Q6 every 5000 steps
        if step % 10000 == 0:
            neighbors = find_neighbors(x, y, z, L, rcut)
            _, Q6g = compute_Q6(x, y, z, neighbors, L)
            Q6_history.append(Q6g)
            step_history.append(step)
            print(f"Step {step}: Q6 = {Q6g:.4f}")

    Q6_all[densities[density_idx]] = np.array(Q6_history)
    step_all[densities[density_idx]] = np.array(step_history)

    mean, err = block_average_error(Q6_history, block_size=5)
    Q6_means.append(mean)
    Q6_errors.append(err)
    print(f"Density {densities[density_idx]:.2f} → Q6 = {mean:.4f} ± {err:.4f}")

# ================================
# Plot Q6 vs MC steps
# ================================
plt.figure(figsize=(8,5))
for density, color in zip(densities, colors):
    plt.plot(step_all[density], Q6_all[density], marker='o', color=color, label=f'η={density:.2f}')

plt.axhline(0.574, ls='--', c='k', label='FCC')
plt.axhline(0.30, ls='--', c='gray', label='Liquid')
plt.xlabel("MC steps")
plt.ylabel("Global $Q_6$")
plt.title("Bond-orientational order vs MC steps")
plt.legend()
plt.tight_layout()
plt.savefig("Q6_LJ.png")
plt.show()

# ================================
# Plot Q6 vs density with error bars
# ================================
plt.figure(figsize=(7,5))
plt.errorbar(densities, Q6_means, yerr=Q6_errors, fmt='o', capsize=4, color='blue')
plt.axhline(0.574, ls='--', c='k', label='FCC')
plt.axhline(0.30, ls='--', c='gray', label='Liquid')
plt.xlabel("Density η")
plt.ylabel("Global $Q_6$")
plt.title("Bond-orientational order vs density")
plt.legend()
plt.tight_layout()
plt.show()
