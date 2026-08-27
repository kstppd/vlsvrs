import vlsvrs
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import numpy as np

file = sys.argv[1]
cid = int(sys.argv[2])
f = vlsvrs.VlsvFile(file)
m_p = 1.6726219e-27
q_e = 1.6021766e-19

sampling_file = "./lucky_particles.txt"
sampled_ptrs = np.loadtxt(sampling_file, delimiter=",")
p_vx, p_vy, p_vz = sampled_ptrs[:, 4], sampled_ptrs[:, 5], sampled_ptrs[:, 6]
v_sq = p_vx**2 + p_vy**2 + p_vz**2
E_total = (0.5 * m_p * v_sq) / q_e
vdf_data = f.read_vdf(cid, "proton")
vdf = vdf_data.squeeze()
nx, ny, nz = vdf.shape
ext = f.get_vspace_mesh_extents("proton")
fig = plt.figure(figsize=(18, 28))
gs = fig.add_gridspec(5, 6, hspace=0.45, wspace=0.6, height_ratios=[1, 1, 1, 0.6, 0.6])

plot_configs = [
    (vdf[:, :, nz//2].T, p_vx, p_vy, ['$v_x$', '$v_y$'], [ext[0], ext[3], ext[1], ext[4]]),
    (vdf[:, ny//2, :].T, p_vx, p_vz, ['$v_x$', '$v_z$'], [ext[0], ext[3], ext[2], ext[5]]),
    (vdf[nx//2, :, :].T, p_vy, p_vz, ['$v_y$', '$v_z$'], [ext[1], ext[4], ext[2], ext[5]]),
]

for i, (vdf_slice, px, py, labels, extent) in enumerate(plot_configs):
    ax_left = fig.add_subplot(gs[i, 0:3])
    v_max = 1e5
    v_min = 1e-16
    safe_slice = np.clip(vdf_slice, v_min, v_max)
    im1 = ax_left.imshow(safe_slice, norm=LogNorm(vmin=v_min, vmax=v_max), 
                         extent=extent, origin='lower', cmap='magma')
    ax_left.set_title(f"VLSV Slice: {labels[0]}-{labels[1]}", fontsize=14)
    ax_left.set_xlabel(f"{labels[0]} [m/s]")
    ax_left.set_ylabel(f"{labels[1]} [m/s]")
    plt.colorbar(im1, ax=ax_left, fraction=0.046, pad=0.04)
    ax_right = fig.add_subplot(gs[i, 3:6])
    counts, xedges, yedges, im2 = ax_right.hist2d(
        px, py, bins=100, range=[[extent[0], extent[1]], [extent[2], extent[3]]], 
        norm=LogNorm(), cmap='magma', cmin=1
    )
    if im2 is not None:
        plt.colorbar(im2, ax=ax_right, fraction=0.046, pad=0.04)
    ax_right.set_title(f"Reconstructed: {labels[0]}-{labels[1]}", fontsize=14)
    ax_right.set_xlabel(f"{labels[0]} [m/s]")
    ax_right.set_ylabel(f"{labels[1]} [m/s]")

vel_axes = [fig.add_subplot(gs[3, 0:2]), fig.add_subplot(gs[3, 2:4]), fig.add_subplot(gs[3, 4:6])]
vel_data = [p_vx*1e-3, p_vy*1e-3, p_vz*1e-3]
vel_labels = ['$v_x$', '$v_y$', '$v_z$']
colors = ['tab:blue', 'tab:orange', 'tab:green']

for ax, data, lbl, clr in zip(vel_axes, vel_data, vel_labels, colors):
    ax.hist(data, bins=80, color=clr, alpha=0.7, edgecolor='black', lw=0.5)
    ax.set_title(f"{lbl} Distribution")
    ax.set_xlabel(f"{lbl} [km/s]")

ax_en = fig.add_subplot(gs[4, :])
ax_en.hist(E_total, bins=150, color='tab:red', alpha=0.8, edgecolor='black', lw=0.5)
ax_en.set_title("Total Particle Energy Distribution")
ax_en.set_xlabel("Energy [eV]")
ax_en.set_yscale('log')
plt.savefig("vdf_full_comparison.png", dpi=300, bbox_inches='tight')
