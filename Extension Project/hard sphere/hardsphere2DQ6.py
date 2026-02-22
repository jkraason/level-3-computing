import numpy as np
import random
import matplotlib.pyplot as plt

# ================================
# Hexagonal lattice
# ================================
def create_hexagonal_lattice(N, r, L):
    spacing = 2.0 * r * 1.01
    dx = spacing
    dy = spacing * np.sqrt(3) / 2

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
# Hard-sphere MC move
# ================================
def mc_move(x, y, r, d, L):
    N = len(x)
    i = np.random.randint(N)

    old_x, old_y = x[i], y[i]

    dx = d * (random.random() - 0.5)
    dy = d * (random.random() - 0.5)

    x[i] = (x[i] + dx) % L
    y[i] = (y[i] + dy) % L

    dx_all = x[i] - x
    dy_all = y[i] - y

    dx_all -= L * np.round(dx_all / L)
    dy_all -= L * np.round(dy_all / L)

    dist2 = dx_all**2 + dy_all**2
    dist2[i] = np.inf

    if np.any(dist2 < (2*r)**2):
        x[i], y[i] = old_x, old_y
        return x, y, False

    return x, y, True


# ================================
# Compute Psi6
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

    psi6_global = np.abs(np.sum(psi6_local)) / N
    return psi6_global


# ================================
# Simulation parameters
# ================================
r = 0.03
d = 0.02
N = 100                    # larger system → smoother Psi6
densities = np.array([0.65,0.68,0.71,0.74])

sample_interval = 10000      # sample less frequently
n_steps = 200000
n_snapshots = 100            # independent configs per sample
colors = ['red','blue','green','orange','purple','brown','black']

plt.figure(figsize=(8,5))

# ================================
# Loop over densities
# ================================
plt.figure(figsize=(8,5))

n_runs = 3   # independent simulations

# ================================
# Loop over densities
# ================================
for idx, density in enumerate(densities):

    L = np.sqrt((N*np.pi*(2*r)**2)/(4*density))

    step_points = np.arange(0, n_steps + 1, sample_interval)
    psi6_runs = np.zeros((n_runs, len(step_points)))

    print(f"\n==== Density η={density:.2f} ====")

    # ====================================
    # Independent runs
    # ====================================
    for run in range(n_runs):

        x, y, spacing = create_hexagonal_lattice(N, r, L)
        rcut_psi6 = 1.5 * spacing

        d_run = d  # independent step size per run

        accepted_moves = 0
        attempted_moves = 0

        sample_index = 0

        # --- Initial Psi6 ---
        psi6_runs[run, sample_index] = compute_psi6_2D(x, y, L, rcut_psi6)
        sample_index += 1

        # ====================================
        # MC simulation
        # ====================================
        for step in range(1, n_steps + 1):

            x, y, accepted = mc_move(x, y, r, d_run, L)

            attempted_moves += 1
            if accepted:
                accepted_moves += 1

            # --------------------------------
            # Acceptance tuning
            # --------------------------------
            if step % 10000 == 0:

                acc_ratio = accepted_moves / attempted_moves

                if acc_ratio > 0.5:
                    d_run *= 1.05
                elif acc_ratio < 0.25:
                    d_run *= 0.95

                accepted_moves = 0
                attempted_moves = 0

            # --------------------------------
            # Sampling
            # --------------------------------
            if step % sample_interval == 0:

                psi6_runs[run, sample_index] = compute_psi6_2D(
                    x, y, L, rcut_psi6
                )
                sample_index += 1

    # ====================================
    # Average across runs
    # ====================================
    psi6_mean = np.mean(psi6_runs, axis=0)
    psi6_sem  = np.std(psi6_runs, axis=0, ddof=1) / np.sqrt(n_runs)

    for s, m, e in zip(step_points, psi6_mean, psi6_sem):
        print(f"η={density:.2f} | step={s} | Psi6={m:.3f} ± {e:.3f}")

    plt.errorbar(step_points,
                 psi6_mean,
                 yerr=psi6_sem,
                 fmt='o-',
                 capsize=3,
                 color=colors[idx],
                 label=fr'$\eta={density:.2f}$')

# Get current x-limits (after plotting your data)
xmin, xmax = plt.xlim()

# Shade hexatic region
plt.fill_between(
    [xmin, xmax],
    0.4,
    0.7,
    color='grey',
    alpha=0.2,
    label='Hexatic phase'
)

# Keep reference lines if you want
plt.axhline(1, color='black', linestyle='--', label='Hexagonal lattice')
plt.axhline(0.2, color='blue', linestyle='--', label='Liquid phase')
plt.xlabel("MC steps", fontsize = 18)
plt.ylabel(r"$\Psi_6$", fontsize = 18)
plt.tick_params(axis='both', labelsize=14)
plt.title("2D Hard-Sphere Bond-Orientational Order")
plt.legend(ncols = 2, fontsize=18)
plt.tight_layout()
plt.savefig("HS2Dpsi6.png")
plt.show()
