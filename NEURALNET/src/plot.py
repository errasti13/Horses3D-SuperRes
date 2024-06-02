import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import os

def plot_error(srcnn, srgan):
    """Plot the error comparison between SRCNN and SRGAN models."""
    time_srcnn = np.linspace(5, 5 + len(srcnn) * 0.002, len(srcnn))
    time_srgan = np.linspace(5, 5 + len(srgan) * 0.002, len(srgan))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(time_srcnn, srcnn, label='SRCNN', linestyle='-', marker='o', markersize=3, color='blue')
    ax.semilogy(time_srgan, srgan, label='SRGAN', linestyle='--', marker='s', markersize=3, color='red')

    ax.axvline(x=5.2, color='black', linestyle=':', linewidth=2, label='Training / New Data Split')

    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Error', fontsize=14, fontweight='bold')
    ax.set_title('SRCNN vs SRGAN', fontsize=16, fontweight='bold')

    ax.legend(loc='upper left', fontsize='large', frameon=True, shadow=True, fancybox=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    fig.tight_layout()

    file_path = "NEURALNET/figures/Error.png"
    try:
        plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Figure saved successfully at {file_path}")
    except Exception as e:
        print(f"Error occurred while saving Figure: {e}")

    plt.show()


def plot_reconstructed(mesh, Q_HO_sol, Q_HO_SRCNN, Q_HO_SRGAN, idx, component='rhou'):
    """Plot the reconstructed flow fields for a given component."""
    component_map = {'rhou': 0, 'rhov': 1, 'rhow': 2}

    if component not in component_map:
        raise ValueError(f"Invalid component '{component}'. Valid options are 'rhou', 'rhov', or 'rhow'.")

    comp_idx = component_map[component]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})

    titles = ['Original', 'SRCNN', 'SRGAN']
    datasets = [Q_HO_sol, Q_HO_SRCNN[idx], Q_HO_SRGAN[idx]]

    mesh_flat = mesh.reshape(-1, 3)
    x, y, z = mesh_flat[:, 0], mesh_flat[:, 1], mesh_flat[:, 2]

    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        data_flat = dataset[..., comp_idx].reshape(-1)
        scatter = ax.scatter(x, y, z, c=data_flat, cmap='jet')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[i])
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_zlim([z.min(), z.max()])
        ax.set_aspect('auto')

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), label=component, orientation='horizontal', fraction=0.05, pad=0.1)

    file_dir = f"NEURALNET/figures/{component}"
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, f"fullFlowField_{idx}.png")

    try:
        plt.savefig(file_path, format='png')
        plt.close(fig)
        print(f"Figure {idx} saved to '{file_path}' successfully!")
    except Exception as e:
        print(f"Error occurred while saving Figure {idx}: {e}")
        plt.close(fig)

    return

def create_gif_from_images(input_dir, output_path, file_pattern='fullFlowField_*.png', duration=0.5):
    """Create an animated GIF from a series of PNG images."""
    try:
        file_pattern = f'{input_dir}/{file_pattern}'
        file_paths = sorted(glob.glob(file_pattern))
        
        images = [imageio.imread(file_path) for file_path in file_paths]

        imageio.mimsave(output_path, images, duration=duration)
        print(f"Animated GIF saved: {output_path}")
    except Exception as e:
        print(f"Error occurred while creating GIF: {e}")
