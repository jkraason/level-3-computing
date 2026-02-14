# ================================
# Imports
# ================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

# ================================
# FCC lattice
# ================================
def fcc_lattice(N, r, L):

    n_cells = int(round((N / 4)**(1/3)))
    a = L / n_cells
    spacing = a / np.sqrt(2)

    basis = np.array([[0,0,0],
                      [0.5,0.5,0],
                      [0.5,0,0.5],
                      [0,0.5,0.5]])

    coords = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    coords.append(a*(np.array([i,j,k]) + b))

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
# Global Q6
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

    return x, y, z


# ================================
# Parameters
# ================================
r = 0.03
d = 0.001
Nx = Ny = Nz = 3
N = 4*Nx*Ny*Nz

densities = np.array([0.49,0.51,0.53,0.55])
L_values = ((N*np.pi*(2*r)**3)/(6*densities))**(1/3)

n_runs = 3
n_steps = 200000
sample_interval = 10000

# ================================
# MAIN LOOP
# ================================
plt.figure(figsize=(8,5))

for density_idx, L in enumerate(L_values):

    print(f"\n=== Density η = {densities[density_idx]:.2f} ===")

    step_points = np.arange(0, n_steps, sample_interval)
    Q6_runs = np.zeros((n_runs, len(step_points)))

    for run in range(n_runs):

        x, y, z, spacing = fcc_lattice(N, r, L)
        rcut = 1.3 * spacing

        sample_index = 0

        for step in range(n_steps):

            x, y, z = mc_move(x, y, z, r, d, L)

            if step % sample_interval == 0:
                neighbors = find_neighbors(x, y, z, L, rcut)
                Q6_runs[run, sample_index] = compute_Q6(x, y, z, neighbors, L)
                sample_index += 1

    # Mean and SEM at EACH step
    Q6_mean = np.mean(Q6_runs, axis=0)
    Q6_sem  = np.std(Q6_runs, axis=0, ddof=1) / np.sqrt(n_runs)

    plt.errorbar(step_points,
                 Q6_mean,
                 yerr=Q6_sem,
                 fmt='o-',
                 capsize=3,
                 label=f'η={densities[density_idx]:.2f}')


plt.axhline(0.574, ls='--', c='k', label='FCC')
plt.axhline(0.30, ls='--', c='gray', label='Liquid')

plt.xlabel("MC steps")
plt.ylabel("Global $Q_6$")
plt.title("Bond-orientational order vs MC steps")
plt.legend()
plt.tight_layout()
plt.savefig("Q6_hs.png")
plt.show()



