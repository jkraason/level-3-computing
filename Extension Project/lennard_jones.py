import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

def create_hexagonal_lattice(N, r, L):
    
    # Hexagonal lattice spacing
    # For hard disks, minimum spacing is 2*r (touching)
    spacing = 2.01 * r  # 1% buffer to ensure no overlaps
    
    # Hexagonal lattice geometry
    dx = spacing
    dy = spacing 
    
    # Calculate how many particles fit
    Nx = int(L / dx)
    Ny = int(L / dy)

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
    
    return x, y, spacing

def mc_move(x, y, r, d, L):
    N = len(x)
    r = 0.03

    # Pick a random particle
    i = np.random.randint(N)
    old_x = x[i]
    old_y = y[i]

    # Save old position
    old_x = x[i]
    old_y = y[i]

    # Propose move
    dx = d * (random.random() - 0.5)
    dy = d * (random.random() - 0.5)
    
    # Apply move with periodic boundaries
    new_x = (x[i] + dx) % L
    new_y = (y[i] + dy) % L

    # Temporarily update position for overlap check
    x[i] = new_x
    y[i] = new_y

    # Compute distances to all other particles with periodic boundaries
    dx_all = x[i] - x
    dx_all -= L * np.round(dx_all / L)

    dy_all = y[i] - y
    dy_all -= L * np.round(dy_all / L)

    dist2 = dx_all**2 + dy_all**2
    r_sum2 = 4*(r)**2
    dist2[i] = np.inf  # ignore self
