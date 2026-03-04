import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==================================================
# Hexagonal lattice
# ==================================================
def create_hexagonal_lattice(N, r, L):
    spacing = 2.0 * r * 1.01
    dx = spacing
    dy = spacing * np.sqrt(3)/2

    Nx = int(L/dx)
    Ny = int(L/dy)

    x, y = [], []
    for i in range(Ny):
        for j in range(Nx):
            x_pos = j*dx + (i%2)*(dx/2)
            y_pos = i*dy
            x_pos %= L
            y_pos %= L
            x.append(x_pos)
            y.append(y_pos)
            if len(x) >= N:
                break
        if len(x) >= N:
            break

    x = np.array(x[:N])
    y = np.array(y[:N])
    # center in box
    x -= (x.max() + x.min())/2
    y -= (y.max() + y.min())/2
    x = (x + L/2) % L
    y = (y + L/2) % L

    return x, y, spacing

# ==================================================
# Hard-sphere Monte Carlo move
# ==================================================
def mc_move(x, y, r, d, L):
    N = len(x)
    i = np.random.randint(N)
    old_x, old_y = x[i], y[i]

    dx = d*(random.random() - 0.5)
    dy = d*(random.random() - 0.5)

    x[i] = (x[i] + dx) % L
    y[i] = (y[i] + dy) % L

    dx_all = x[i] - x
    dy_all = y[i] - y
    dx_all -= L * np.round(dx_all/L)
    dy_all -= L * np.round(dy_all/L)

    dist2 = dx_all**2 + dy_all**2
    dist2[i] = np.inf

    if np.any(dist2 < (2*r)**2):
        x[i], y[i] = old_x, old_y
        return x, y, False

    return x, y, True

# ==================================================
# Radial Distribution Function
# ==================================================
def pairCorrelationFunction_2D(x, y, L, rMax, dr):
    N = len(x)
    rho = N / L**2
    r_array = np.arange(dr/2, rMax, dr)
    g_r = np.zeros(len(r_array))

    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dx -= L * np.round(dx/L)
            dy -= L * np.round(dy/L)
            r_val = np.sqrt(dx**2 + dy**2)
            if r_val < rMax and r_val > 0:
                bin_idx = int(r_val/dr)
                if bin_idx < len(g_r):
                    g_r[bin_idx] += 2

    for i, r_val in enumerate(r_array):
        shell_area = 2*np.pi*r_val*dr
        g_r[i] /= (rho * shell_area * N)
    return r_array, g_r

def averaged_g_r(x, y,r, d, L, rMax, dr, num_sims, sample, equil_steps=0, block_size=5):
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

# ==================================================
# Simulation parameters
# ==================================================
os.makedirs("frames", exist_ok=True)

r = 0.03
d_initial = 0.01
dr = 0.1*r
Nx = 20
Ny = 20
N=Nx*Ny

densities = np.array([0.65,0.68, 0.71,0.74])
L_values= np.sqrt((N*np.pi*(2*r)**2)/(4*densities))
colors = ['red','blue','green','orange','purple','brown','black']

plt.figure(figsize=(8,6))

# ==================================================
# Main loop over densities
# ==================================================
for idx, (density, L) in enumerate(zip(densities, L_values)):
    gr_file = f"N=400gr_HS2D_eta_{density:.2f}.npz"

    if os.path.exists(gr_file):
        print(f"Found existing data for η={density:.2f}. Loading instead of recomputing.")
        data = np.load(gr_file)
        r_vals = data["r_vals"]
        g_mean = data["g_mean"]
        g_err = data["g_err"]
    else:
        print(f"\n=== Density {density:.2f} ===")
        accepted_moves = 0
        eq_frame_index = 0
        x, y, spacing = create_hexagonal_lattice(N, r, L)
        d = d_initial

        frames_dir = f"frames_eta_{density:.2f}"
        os.makedirs(frames_dir, exist_ok=True)
        frame_index = 0

        # MC simulation + frame saving + tuning
        tune_steps = 50000
        tune_interval = 10000
        accepted_moves = 0
        attempted_moves = 0

        for step in range(1000000):
            x, y, accepted = mc_move(x, y, r, d, L)
            attempted_moves += 1
            if accepted:
                accepted_moves += 1

            # Save frames every 10k steps
            if step % 10000 == 0:
                fig, ax = plt.subplots(figsize=(6,6))
                for j in range(N):
                    ax.add_patch(plt.Circle((x[j],y[j]), r, color='blue', ec='black', alpha=0.8))
                ax.set_xlim(0,L)
                ax.set_ylim(0,L)
                ax.set_aspect('equal')
                ax.set_title(f'η={density:.2f}, step={step}', fontsize=18)
                ax.tick_params(axis='both', labelsize=16)
                plt.tight_layout()
                plt.savefig(f"{frames_dir}/frame_{frame_index:04d}.png", dpi=150)
                plt.close(fig)
                frame_index += 1

            # Tune displacement
        acceptance_ratio = accepted_moves/attempted_moves
        while acceptance_ratio>0.5 or acceptance_ratio<0.25:
            if acceptance_ratio>0.5:
                accepted_moves=0
                d = d*1.05 
                print("d=", d)
                for j in range(0,10000):
                    x,y,z,accepted = mc_move(x,y,r,d,L)
                    if accepted:
                        accepted_moves+=1
                acceptance_ratio = accepted_moves/10000
                print("Acceptance ratio (large):", acceptance_ratio)
            if acceptance_ratio<0.25:
                accepted_moves = 0
                d = d*0.95
                print("d =",d)
                for j in range(0,10000):
                    x,y,z,accepted = mc_move(x,y,r,d,L)
                    if accepted:
                        accepted_moves+=1
                acceptance_ratio = accepted_moves/10000
                print("Acceptance ratio (small):", acceptance_ratio)
        print("final acceptance ratio:", acceptance_ratio)

        print(f"Final d={d:.5f}")

        # Build GIF for this density
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        images = [Image.open(os.path.join(frames_dir,f)) for f in frame_files]
        if images:
            gif_path = f"equilibration_eta_{density:.2f}.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
            print(f"Saved GIF: {gif_path}")

        # RDF production run
        
        r_vals, g_r, g_err = averaged_g_r(x, y, r, d, L, rMax=10*r, dr=dr, num_sims = 500000, sample = 1000, equil_steps = 0, block_size = 5)
                # =====================================
        # SAVE AVERAGED g(r) FOR THIS DENSITY
        # =====================================

        gr_file = f"N=400gr_HS2D_eta_{density:.2f}.npz"

        np.savez_compressed(
            gr_file,
            r_vals=r_vals,
            g_mean=g_r,
            g_err=g_err,
            density=density,
            N=N,
            L=L,
            sigma=2*r,
            dr=dr,
            rMax=10*r
        )

        print(f"Saved averaged g(r) to {gr_file}")
    plt.errorbar(r_vals/(2*r), g_mean, yerr = g_err, linewidth=2, color=colors[idx], label=f'η={density:.2f}')

# ==================================================
# Final RDF plot
# ==================================================
# Final RDF plot
plt.axhline(1, color='black', linestyle='--', label = 'Ideal gas')
plt.xlabel('r/σ', fontsize=18)
plt.ylabel('g(r)', fontsize=18)
plt.tick_params(axis='both', labelsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("N=400HS2Dgr.png", dpi=300)
plt.show()


print("Simulation complete.")
