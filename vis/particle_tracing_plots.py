# import numpy as np
import sys
sys.path.append("/wrk-vakka/users/souhadah/vlsvrs/vis/")

import vlsvrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ptrReader
import glob
import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import glob,os

RE = 6_378_137.0
EV_TO_J = 1.602176634e-19
PROTON_MASS = 1.67262192369e-27

run = "AID"


def plot_particles_over_efield(
    ptr_file: str,
    vlsv_file: str,
    *,
    plane: str = "XY",
    e_component: int = 1,
    e_scale: float = 1e3,
    e_vmin: float = -3.0,
    e_vmax: float =  3.0,
    part_energy_vmin: float = 0.0,
    part_energy_vmax: float = 6.0,
    cmap_field: str = "PRGn",
    cmap_particles: str = "seismic",
    title_time_s: float | None = None,
    save_path: str | None = None,
    dpi: int = 300,
):
    print(ptr_file,vlsv_file)
    f = vlsvrs.VlsvFile(vlsv_file)
    Efield = f.read_variable_f32("vg_e_vol", op=0).squeeze()
    x_min, y_min, z_min, x_max, y_max, z_max = f.get_spatial_mesh_extents()
    if plane.upper() == "XY":
        extent = [x_min/RE, x_max/RE, y_min/RE, y_max/RE]
        ef_slice = Efield[:, :, e_component].T * e_scale
    elif plane.upper() == "XZ":
        extent = [x_min/RE, x_max/RE, z_min/RE, z_max/RE]
        ef_slice = Efield[:, :, e_component].T * e_scale
    else:
        raise ValueError("plane must be 'XY' or 'XZ'")

    x, y, z, vx, vy, vz = ptrReader.read_ptr2_file(ptr_file)
    xi, yi, zi, vxi, vyi, vzi = ptrReader.read_ptr2_file(f'/wrk-vakka/users/souhadah/vlsvrs/{run}_sampled_box/state.0000000.ptr')

    xRE, yRE, zRE = x/RE, y/RE, z/RE
    xiRE, yiRE, ziRE = xi/RE, yi/RE, zi/RE

    # store = 0.5 * PROTON_MASS * (vx**2 + vz**2) / EV_TO_J * 1e-3
    particle_energy_keV = 0.5 * PROTON_MASS * (vx**2 + vz**2) / EV_TO_J * 1e-3
    init_energies = 0.5 * PROTON_MASS * (vxi**2 + vzi**2) / EV_TO_J * 1e-3

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # fig.patch.set_facecolor("black")
    # ax.set_facecolor("black")

    im = ax.imshow(
        ef_slice,
        origin="lower",
        extent=extent,
        cmap=cmap_field,
        vmin=e_vmin, vmax=e_vmax,
        interpolation="nearest",
        alpha=1
    )

    cbar = fig.colorbar(im, ax=ax, label=r"$E_{}$ (mV/m)".format(["x","y","z"][e_component]))
    # cbar.outline.set_edgecolor("white")
    # cbar.ax.yaxis.set_tick_params(color="white")
    # cbar.ax.tick_params(colors="white")
    cbar.set_label(cbar.ax.get_ylabel(), color="k")

    if plane.upper() == "XY":
        sc = ax.scatter(
            xRE, yRE,
            s=0.15,
            c=(particle_energy_keV - init_energies) / init_energies,
            cmap=cmap_particles,
            alpha=1,
            vmin=-1,vmax=1
        )
        cbar2 = fig.colorbar(sc, ax=ax ,label="$\delta E_{kinetic}$")
        ax.set_xlabel("X [R$_E$]")
        ax.set_ylabel("Y [R$_E$]")
    else:
        sc = ax.scatter(
            xRE, zRE,
            s=0.15,
            c=(particle_energy_keV - init_energies) / init_energies,
            alpha=1,
            cmap=cmap_particles
        )
        cbar2 = fig.colorbar(sc, ax=ax ,label="Particle kinetic energy gain/loss (keV)")
        ax.set_xlabel("X [R$_E$]")
        ax.set_ylabel("Z [R$_E$]")

    # cbar2.outline.set_edgecolor("white")
    # cbar2.ax.yaxis.set_tick_params(color="white")
    # cbar2.ax.tick_params(colors="white")
    cbar2.set_label(cbar2.ax.get_ylabel(), color="k")

    title = f"{run} GC PTR"
    if title_time_s is not None:
        title += f" — t = {title_time_s:.1f} s"
    ax.set_title(title, color="k")

    # for spine in ax.spines.values():
    #     spine.set_color("#CCCCCC")
    #     spine.set_linewidth(0.8)

    # # ax.tick_params(colors="white", which="both")
    # ax.xaxis.label.set_color("white")
    # ax.yaxis.label.set_color("white")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # ax_hist = ax.inset_axes([0.05, -0.45, 0.9, 0.25], facecolor='#FFFFFF40')
    # ax_hist.hist(
    #     store,
    #     bins=50,
    #     color='cyan',
    #     alpha=0.7,
    #     density=True,
    #     log=True,
    #     edgecolor='black',
    # )

    # ax_hist.hist(
    #     init_energies,
    #     bins=50,
    #     color='magenta',
    #     histtype='step',
    #     linewidth=1.5,
    #     density=True,
    #     log=True,
    # )


    # ax_hist.set_title("Population Energy (keV)", color='white', fontsize=8)
    # ax_hist.tick_params(axis='x', colors='white', labelsize=7)
    # ax_hist.tick_params(axis='y', colors='white', labelsize=7)
    # ax_hist.set_yticks([])
    # ax_hist.set_xlim(0, 150)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig, ax


PTR_DIR = f"/wrk-vakka/users/souhadah/vlsvrs/{run}_sampled_box/"
VLSV_DIR = f"/wrk-vakka/group/spacephysics/vlasiator/2D/{run}/bulk/"
OUTPUT_DIR = f"/wrk-vakka/users/souhadah/{run}/particle_tracing/VDF_sampled_box/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
ptr_files = sorted(glob.glob(os.path.join(PTR_DIR, "state.*.ptr")))
vlsv_files = sorted(glob.glob(os.path.join(VLSV_DIR, "bulk.*.vlsv")))
x, y, z, vx, vy, vz = ptrReader.read_ptr2_file(ptr_files[0])
init_en = 0.5 * PROTON_MASS * (vx**2 + vz**2) / EV_TO_J * 1e-3

def generate_plot(idx):
    try:
        ptr_file = ptr_files[idx]
        vlsv_file = vlsv_files[10:][idx]
        fig, ax = plot_particles_over_efield(
            ptr_file,
            vlsv_file,
            plane="XY",
            e_component=1,
            e_vmin=-3, e_vmax=3,
            part_energy_vmin=0, part_energy_vmax=30,
            title_time_s=None
        )
        output_filename = os.path.join(OUTPUT_DIR, f"plot_{idx:04d}.png")
        plt.savefig(output_filename, dpi=300)
        plt.close(fig)

        return f"SUCCESS: Saved {output_filename}"

    except Exception as e:
        return f"ERROR processing index {idx}: {e}"

if __name__ == "__main__":
    indices = range(len(ptr_files))

    num_processes = os.cpu_count()
    print(f"Starting parallel processing with {num_processes} cores for {len(indices)} files.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_plot, indices)

    for res in results:
        print(res)

    print("\nAll tasks completed.")
