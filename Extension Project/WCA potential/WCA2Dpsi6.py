import numpy as np
import random
import matplotlib.pyplot as plt
import os
# ================================
# System Parameters
# ================================
N = 100                 # number of particles
r = 0.03
sigma = 2 * r
epsilon = 1.0

kB = 1.0
T = 1.0
beta = 1.0 / (kB * T)

densities = np.array([0.68,0.71,0.74,0.77])
colors = ['red', 'blue', 'green', 'orange', 'purple']

initial_d = 0.03
n_steps = 200_000
sample_interval = 10_000
n_runs = 3   # independent simulations

# ================================
# Hexagonal lattice
# ================================
def create_hexagonal_lattice(N, r, L):
    spacing = 2.0 * r * 1.01
    dx = spacing
    dy = spacing * np.sqrt(3)/2

    x = []
    y = []
    row = 0

    while len(x) < N:
        shift = 0 if row % 2 == 0 else dx / 2
        col = 0
        while True:
            x_pos = col * dx + shift
            y_pos = row * dy
            if x_pos >= L:
                break
            x.append(x_pos % L)
            y.append(y_pos % L)
            if len(x) >= N:
                break
            col += 1
        row += 1

    return np.array(x), np.array(y), spacing

# ================================
# Lennard-Jones MC
# ================================
def wca_potential(r, sigma, epsilon=1.0):
    rcut = 2**(1/6) * sigma
    
    if r >= rcut:
        return 0.0
    
    sr6 = (sigma / r)**6
    return 4 * epsilon * (sr6**2 - sr6) + epsilon

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
        r_old = np.sqrt(dx**2 + dy**2)

        dx = new_x - x[j]
        dy = new_y - y[j]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        r_new = np.sqrt(dx**2 + dy**2)

        if r_old < rcut:
            dU -= wca_potential(r_old, sigma, rcut)
        if r_new < rcut:
            dU += wca_potential(r_new, sigma, rcut)

    if dU <= 0 or random.random() < np.exp(-beta*dU):
        x[i], y[i] = new_x, new_y
        return True
    return False

# ================================
# Psi6
# ================================
def compute_psi6_2D(x, y, L, rcut):
    N = len(x)
    psi6_local = np.zeros(N, dtype=complex)

    for i in range(N):
        neighbors = []
        for j in range(N):
            if i == j:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dx -= L * np.round(dx / L)
            dy -= L * np.round(dy / L)
            r_ij = np.sqrt(dx**2 + dy**2)
            if r_ij < rcut:
                neighbors.append((dx, dy))
        if len(neighbors) == 0:
            continue
        sum_bonds = 0
        for dx, dy in neighbors:
            theta = np.arctan2(dy, dx)
            sum_bonds += np.exp(1j * 6 * theta)
        psi6_local[i] = sum_bonds / len(neighbors)

    return np.abs(np.sum(psi6_local)) / N

# ================================
# Main Simulation
# ================================
plt.figure(figsize=(8,5))

for idx, density in enumerate(densities):
    psi_file = f"WCA2Dpsi6_eta_{density:.2f}.npz"

    if os.path.exists(psi_file):
        print(f"Found existing data for η={density:.2f}. Loading instead of recomputing.")
        data = np.load(psi_file)

        step_points = data["step_points"]
        psi6_mean = data["psi6_mean"]
        psi6_sem  = data["psi6_sem"]

        plt.errorbar(step_points,
                    psi6_mean,
                    yerr=psi6_sem,
                    fmt='o-',
                    capsize=3,
                    color=colors[idx],
                    label=fr'$\eta={density:.2f}$')

        continue
    else: 
        L = np.sqrt((N * np.pi * sigma**2) / (4 * density))
        d = initial_d
        rcut = 2.5 * sigma

        step_points = np.arange(0, n_steps + 1, sample_interval)
        psi6_runs = np.zeros((n_runs, len(step_points)))

        print(f"\n==== Density η={density:.3f} ====")

        for run in range(n_runs):
            x, y, spacing = create_hexagonal_lattice(N, r, L)
            rcut_psi6 = 1.5 * spacing

            sample_index = 0
            psi6_runs[run, sample_index] = compute_psi6_2D(x, y, L, rcut_psi6)
            sample_index += 1

            accepted_moves = 0
            attempted_moves = 0
            d_run = d

            for step in range(1, n_steps + 1):
                accepted = mc_move_lj(x, y, d_run, L, sigma, rcut, beta)

                attempted_moves += 1
                if accepted:
                    accepted_moves += 1

                # Adaptive step size every 10000 moves
                if step % 10000 == 0:
                    acc_ratio = accepted_moves / attempted_moves
                    if acc_ratio > 0.5:
                        d_run *= 1.05
                    elif acc_ratio < 0.25:
                        d_run *= 0.95
                    accepted_moves = 0
                    attempted_moves = 0

                # Sample Psi6
                if step % sample_interval == 0:
                    psi6_runs[run, sample_index] = compute_psi6_2D(x, y, L, rcut_psi6)
                    sample_index += 1

        # Average over independent runs
        psi6_mean = np.mean(psi6_runs, axis=0)
        psi6_sem = np.std(psi6_runs, axis=0, ddof=1) / np.sqrt(n_runs)

        for s, m, e in zip(step_points, psi6_mean, psi6_sem):
            print(f"η={density:.3f} | step={s} | Psi6={m:.3f} ± {e:.3f}")
        
        psi_file = f"WCA2Dpsi6_eta_{density:.2f}.npz"
        np.savez_compressed(
                psi_file,
                step_points=step_points,
                psi6_mean=psi6_mean,
                psi6_sem=psi6_sem,
                density=density,
                N=N,
                L=L,
                r=r,
                n_runs=n_runs,
                n_steps=n_steps,
                sample_interval=sample_interval
            )

        print(f"Saved ψ6 data to {psi_file}")

    plt.errorbar(step_points,
                 psi6_mean,
                 yerr=psi6_sem,
                 fmt='o-',
                 capsize=3,
                 color=colors[idx],
                 label=fr'$\eta={density:.3f}$')

# Shade hexatic region
xmin, xmax = plt.xlim()
plt.fill_between([xmin, xmax], 0.4, 0.7, color='grey', alpha=0.2, label='Hexatic phase')
plt.axhline(1, color='black', linestyle='--', label='Hexagonal lattice')
plt.axhline(0.2, color='blue', linestyle='--', label='Liquid phase')

plt.xlabel("MC steps", fontsize=18)
plt.ylabel(r"$\Psi_6$", fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.legend(ncols=2,fontsize=18)
plt.tight_layout()
plt.savefig("WCA_2Dpsi6.png")
plt.show()