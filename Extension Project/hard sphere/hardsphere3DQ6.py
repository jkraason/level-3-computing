# ================================
# Imports
# ================================
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from PIL import Image
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D

# ================================
# FCC lattice
# ================================
def fcc_lattice(N, r, L, tol=1e-8):
    if N % 4 != 0:
        raise ValueError("FCC lattice requires N multiple of 4")

    n_cells = int(round((N / 4)**(1/3)))
    if 4 * n_cells**3 != N:
        raise ValueError("N must equal 4*n_cells^3")

    a = L / n_cells
    spacing = a / np.sqrt(2)

    if spacing < 2*r - tol:
        raise ValueError("FCC overlap unavoidable")

    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    coords = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    coords.append(a*(np.array([i,j,k])+b))
    coords = np.array(coords)
    return coords[:,0], coords[:,1], coords[:,2], spacing

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

# ================================
# Monte Carlo move
# ================================
def mc_move(x, y, z, r, d, L):
    i = np.random.randint(len(x))
    old = x[i], y[i], z[i]

    dx, dy, dz = d*(np.random.rand(3)-0.5)
    x[i] = (x[i]+dx) % L
    y[i] = (y[i]+dy) % L
    z[i] = (z[i]+dz) % L

    dx_all = x[i]-x
    dy_all = y[i]-y
    dz_all = z[i]-z
    dx_all -= L*np.round(dx_all/L)
    dy_all -= L*np.round(dy_all/L)
    dz_all -= L*np.round(dz_all/L)

    dist2 = dx_all**2 + dy_all**2 + dz_all**2
    dist2[i] = np.inf

    if np.any(dist2 < (2*r)**2):
        x[i], y[i], z[i] = old
        return x, y, z, False

    return x, y, z, True

# ================================
# Block averaging for error
# ================================
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

# ================================
# Simulation parameters
# ================================
r = 0.03
d = 0.001
dr = 0.1*r
rcut = 1.4*(2*r)
Nx = Ny = Nz = 3
N = 4*Nx*Ny*Nz

densities = np.array([0.3,0.4,0.5,0.6])
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

    for step in range(1500000):
        x, y, z, acc = mc_move(x, y, z, r, d, L)
        accepted_moves += acc

        # Compute Q6 every 5000 steps
        if step % 5000 == 0:
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
