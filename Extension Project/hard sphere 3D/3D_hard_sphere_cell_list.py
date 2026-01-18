import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------
# 3D FCC lattice creation
# ----------------------
def create_fcc_lattice(N, r, L):
    a = 2 * r * np.sqrt(2) * 1.01  # FCC lattice constant
    n_cells = int(np.ceil((N / 4)**(1/3)))  # 4 particles per FCC unit cell

    x, y, z = [], [], []
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a

    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                origin = np.array([i, j, k]) * a
                for b in basis:
                    pos = origin + b
                    if len(x) < N:
                        x.append(pos[0] % L)
                        y.append(pos[1] % L)
                        z.append(pos[2] % L)

    x = np.array(x[:N])
    y = np.array(y[:N])
    z = np.array(z[:N])

    # Center lattice
    x = x - (x.max() + x.min())/2 + L/2
    y = y - (y.max() + y.min())/2 + L/2
    z = z - (z.max() + z.min())/2 + L/2

    return x, y, z, a

# ----------------------
# Cell list construction
# ----------------------
def build_cell_list(x, y, z, L, cell_size):
    n_cells = int(L / cell_size)
    cells = [[[[] for _ in range(n_cells)] for _ in range(n_cells)] for _ in range(n_cells)]
    for idx in range(len(x)):
        ci = int(x[idx] / cell_size) % n_cells
        cj = int(y[idx] / cell_size) % n_cells
        ck = int(z[idx] / cell_size) % n_cells
        cells[ci][cj][ck].append(idx)
    return cells, n_cells

# ----------------------
# Monte Carlo move with cell list
# ----------------------
def mc_move_3D_celllist(x, y, z, r, d, L, cells, n_cells, cell_size):
    N = len(x)
    i = np.random.randint(N)
    old_x, old_y, old_z = x[i], y[i], z[i]

    dx = d * (random.random() - 0.5)
    dy = d * (random.random() - 0.5)
    dz = d * (random.random() - 0.5)

    x[i] = (x[i] + dx) % L
    y[i] = (y[i] + dy) % L
    z[i] = (z[i] + dz) % L

    ci = int(x[i] / cell_size) % n_cells
    cj = int(y[i] / cell_size) % n_cells
    ck = int(z[i] / cell_size) % n_cells

    r2 = (2*r)**2
    overlap = False
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                ni = (ci + di) % n_cells
                nj = (cj + dj) % n_cells
                nk = (ck + dk) % n_cells
                for jdx in cells[ni][nj][nk]:
                    if jdx == i:
                        continue
                    dx_ = x[i] - x[jdx]
                    dy_ = y[i] - y[jdx]
                    dz_ = z[i] - z[jdx]
                    dx_ -= L * np.round(dx_ / L)
                    dy_ -= L * np.round(dy_ / L)
                    dz_ -= L * np.round(dz_ / L)
                    if dx_**2 + dy_**2 + dz_**2 < r2:
                        overlap = True
                        break
                if overlap: break
            if overlap: break
        if overlap: break

    if overlap:
        x[i], y[i], z[i] = old_x, old_y, old_z
        return x, y, z, False
    else:
        old_ci = int(old_x / cell_size) % n_cells
        old_cj = int(old_y / cell_size) % n_cells
        old_ck = int(old_z / cell_size) % n_cells
        cells[old_ci][old_cj][old_ck].remove(i)
        cells[ci][cj][ck].append(i)
        return x, y, z, True

# ----------------------
# 3D radial distribution function
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
def averaged_g_r(x, y,z, r, d, L, rMax, dr, num_sims, sample):
    # Initialize variables for averaging
    g_r_sum = None
    sample_count = 0
    
    for i in range(num_sims):
        x, y, accepted = mc_move_3D_celllist(x, y, z, r, d, L)
        
        if i % sample == 0:
            r_array, g_r_current = pairCorrelationFunction_3D(x, y,z, L, rMax, dr)
            
            # Initialize g_r_sum on first sample
            if g_r_sum is None:
                g_r_sum = np.zeros_like(g_r_current)
            
            # Accumulate the current g(r)
            g_r_sum += g_r_current
            sample_count += 1
    
    # Calculate the average
    if sample_count > 0:
        g_r_avg = g_r_sum / sample_count
    else:
        g_r_avg = np.array([])  # or handle the no-samples case appropriately
    
    return r_array, g_r_avg



# ----------------------
# Simulation parameters
# ----------------------
N = 30         # smaller N for speed
r = 0.03
d = 0.00257
dr = 0.1*r
density = 0.5
L1 = N*np.pi*(2*r)**3
L2 = 6*density
L = (L1/L2)**(1/3)
cell_size = 2.5*r

# ----------------------
# Initialize FCC lattice + cell list
# ----------------------
x, y, z, spacing = create_fcc_lattice(N, r, L)
cells, n_cells = build_cell_list(x, y, z, L, cell_size)

# --- Step 1: small random displacement to break perfect lattice ---
x += 0.01*r * (np.random.rand(N) - 0.5)
y += 0.01*r * (np.random.rand(N) - 0.5)
z += 0.01*r * (np.random.rand(N) - 0.5)
# FIX: rebuild cell list
cells, n_cells = build_cell_list(x, y, z, L, cell_size)
# ----------------------
# Step 2: robust dynamic d adjustment
# ----------------------
d_max = 0.1
d_min = 1e-5
max_iterations = 200
target_accept_min = 0.25
target_accept_max = 0.5


melt_steps = 500000  # number of MC steps for melting
accepted_moves = 0

for step in range(melt_steps):
    x, y, z, accepted = mc_move_3D_celllist(x, y, z, r, d, L, cells, n_cells, cell_size)
    if accepted:
        accepted_moves += 1

acceptance_ratio = accepted_moves / melt_steps
print(f"Melting done. Acceptance ratio = {acceptance_ratio:.3f}")


iteration = 0
while iteration < max_iterations:
    accepted_moves = 0
    adjust_steps = 5000
    for step in range(adjust_steps):
        x, y, z, accepted = mc_move_3D_celllist(x, y, z, r, d, L, cells, n_cells, cell_size)
        if accepted: accepted_moves += 1
    acceptance_ratio = accepted_moves / adjust_steps
    print(f"Iteration {iteration+1}, d={d:.5f}, acceptance={acceptance_ratio:.3f}")

    if target_accept_min <= acceptance_ratio <= target_accept_max:
        break
    elif acceptance_ratio > target_accept_max:
        d *= 1.05
        d = min(d, d_max)
    elif acceptance_ratio < target_accept_min:
        d *= 0.95
        if d < d_min:
            d = d_min
            break
    iteration += 1

print(f"Final move amplitude d = {d:.5f}, acceptance ratio = {acceptance_ratio:.3f}")

# ----------------------
# Main MC simulation
# ----------------------
total_steps = 50000
accepted_moves = 0
for step in range(total_steps):
    x, y, z, accepted = mc_move_3D_celllist(x, y, z, r, d, L, cells, n_cells, cell_size)
    if accepted: accepted_moves += 1
print("Final acceptance ratio:", accepted_moves / total_steps)

# ----------------------
# Compute g(r)
# ----------------------
r_vals, g_r = averaged_g_r(x, y, z, r,d, L, 10*r, dr, 1000,10)
plt.figure()
plt.plot(r_vals/(2*r), g_r, linewidth=2)
plt.xlabel('r/σ'); plt.ylabel('g(r)')
plt.axhline(1, linestyle='--', color='black', alpha=0.5)
plt.show()
