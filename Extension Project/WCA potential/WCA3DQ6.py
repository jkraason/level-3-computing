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

densities = np.array([0.49,0.53,0.57,0.61,0.65])
L_values = ((N*np.pi*(2*r)**3)/(6*(densities)))**(1/3)
colors = ['red', 'blue', 'green', 'purple']

dr = 0.1 * r
rcut = 2**(1/6)*sigma

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

def wca_potential(r, sigma, epsilon=1.0):
    rcut = 2**(1/6) * sigma
    
    if r >= rcut:
        return 0.0
    
    sr6 = (sigma / r)**6
    return 4 * epsilon * (sr6**2 - sr6) + epsilon

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
            theta = np.arccos(dz/r)
            phi = np.arctan2(dy, dx)

            for m in range(-l, l+1):
                qlm[i, m+l] += sph_harm(m, l, phi, theta)

        qlm[i] /= len(neighbors[i])

    qlm_global = np.mean(qlm, axis=0)

    Q6 = np.sqrt((4*np.pi/(2*l+1)) *
                 np.sum(np.abs(qlm_global)**2))

    return Q6

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
            dU -= wca_potential(r_old, sigma, rcut)
        if r_new < rcut:
            dU += wca_potential(r_new, sigma, rcut)

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
Nx = Ny = Nz = 3
N = 4*Nx*Ny*Nz

densities = np.array([0.49,0.53,0.57,0.61,0.65])
colors = ['red','blue','green','orange','purple']
L_values = ((N*np.pi*(2*r)**3)/(6*densities))**(1/3)

# ================================
# Storage for Q6
# ================================
Q6_all = {}
step_all = {}
Q6_means = []
Q6_errors = []
n_runs = 3
n_steps = 1000000
sample_interval = 10000
# ================================
# MAIN LOOP
# ================================
# ================================
# MAIN LOOP
# ================================
plt.figure(figsize=(8,5))

for density_idx, L in enumerate(L_values):

    density = densities[density_idx]
    q6_file = f"Q6_eta_{density:.2f}.npz"

    print(f"\n=== Density η = {density:.2f} ===")

    if os.path.exists(q6_file):
        print("Found existing data. Loading instead of recomputing.")
        data = np.load(q6_file)

        step_points = data["step_points"]
        Q6_mean = data["Q6_mean"]
        Q6_sem = data["Q6_sem"]

    else:
        step_points = np.arange(0, n_steps, sample_interval)
        Q6_runs = np.zeros((n_runs, len(step_points)))

        for run in range(n_runs):

            x, y, z, spacing = fcc_lattice(N, r, L)
            rcut = 1.3 * spacing

            sample_index = 0

            for step in range(n_steps):

                x, y, z = mc_move_lj(x, y, z, r, d, L)

                if step % sample_interval == 0:
                    neighbors = find_neighbors(x, y, z, L, rcut)
                    Q6_runs[run, sample_index] = compute_Q6(x, y, z, neighbors, L)
                    sample_index += 1

        # Mean and SEM at EACH step
        Q6_mean = np.mean(Q6_runs, axis=0)
        Q6_sem  = np.std(Q6_runs, axis=0, ddof=1) / np.sqrt(n_runs)

        # =========================
        # SAVE RESULTS
        # =========================
        np.savez_compressed(
            q6_file,
            step_points=step_points,
            Q6_mean=Q6_mean,
            Q6_sem=Q6_sem,
            density=density,
            N=N,
            n_runs=n_runs,
            n_steps=n_steps,
            sample_interval=sample_interval
        )

        print(f"Saved Q6 data to {q6_file}")

    # =========================
    # PLOT (always executed)
    # =========================
    plt.errorbar(step_points,
                 Q6_mean,
                 yerr=Q6_sem,
                 fmt='o-',
                 capsize=3,
                 label=f'η={density:.2f}')


plt.axhline(0.574, ls='--', c='k', label='FCC')
plt.axhline(0.30, ls='--', c='gray', label='Liquid')
plt.xlabel("MC steps", fontsize = 18)
plt.ylabel("Global $Q_6$", fontsize = 18)
plt.tick_params(axis='both', labelsize = 14)
plt.legend(ncols = 2,fontsize = 18)
plt.savefig('Q6LJ3D.png')
plt.tight_layout()
plt.show()