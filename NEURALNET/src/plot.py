import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import glob
import os
import imageio


def plot_error(srcnn, srgan):
    # Calculate time values based on array lengths
    time_srcnn = np.linspace(0, len(srcnn) * 0.002, len(srcnn))
    time_srgan = np.linspace(0, len(srgan) * 0.002, len(srgan))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot errors with semilogy (log-scale) and customize line styles
    ax.semilogy(time_srcnn, srcnn, label='SRCNN', linestyle='-', marker='o', markersize=1)
    ax.semilogy(time_srgan, srgan, label='SRGAN', linestyle='--', marker='s', markersize=1)

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_title('Error Comparison: SRCNN vs SRGAN')

    # Add legend with improved formatting
    ax.legend(loc='upper left', fontsize='medium')

    # Show grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    file_path = f"NEURALNET/figures/Error.png"
    try:
        # Save the figure as a PNG file
        plt.savefig(file_path, format='png')
    except Exception as e:
        print(f"Error occurred while saving Figure: {e}")


def plot_heatmap_and_errors(Q_LO_ind, Q_HO_ind, Q_HO_sol, L2_Error):
    fig, ax = plt.subplots(3, 3)

    im0 = ax[0, 0].matshow(Q_LO_ind[23, 0, :, :, 0], cmap='jet')
    im1 = ax[0, 1].matshow(Q_HO_ind[23, 0, :, :, 0], cmap='jet')
    im2 = ax[0, 2].matshow(Q_HO_sol[48, 23, 0, :, :, 0], cmap='jet')

    im3 = ax[1, 0].matshow(Q_LO_ind[23, 0, :, :, 1], cmap='jet')
    im4 = ax[1, 1].matshow(Q_HO_ind[23, 0, :, :, 1], cmap='jet')
    im5 = ax[1, 2].matshow(Q_HO_sol[48, 23, 0, :, :, 1], cmap='jet')

    im6 = ax[2, 0].matshow(Q_LO_ind[23, 0, :, :, 2], cmap='jet')
    im7 = ax[2, 1].matshow(Q_HO_ind[23, 0, :, :, 2], cmap='jet')
    im8 = ax[2, 2].matshow(Q_HO_sol[48, 23, 0, :, :, 2], cmap='jet')

    cbar2 = fig.colorbar(im2, ax=ax[0, :])
    cbar5 = fig.colorbar(im5, ax=ax[1, :])
    cbar8 = fig.colorbar(im8, ax=ax[2, :])

    fig.text(0.1, 0.8, 'Rhou', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.5, 'Rhov', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.2, 'Rhow', va='center', ha='right', rotation='vertical')

    ax[0, 0].set_title('Original LO')
    ax[0, 1].set_title('Original HO')
    ax[0, 2].set_title('Predicted HO')

    for axes_row in ax:
        for axes in axes_row:
            axes.set_xticks([])
            axes.set_yticks([])

    plt.savefig("NEURALNET/HeatMap.pdf")



def plot_reconstructed(mesh, Q_HO_sol, Q_HO_SRCNN, Q_HO_SRGAN, idx):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})  # Create a figure with three subplots

    scatter1 = axes[0].scatter(mesh.reshape(-1, 3)[:, 0], mesh.reshape(-1, 3)[:, 1], mesh.reshape(-1, 3)[:, 2], 
                               c=Q_HO_sol[..., 0].reshape(-1), cmap='jet')
    scatter2 = axes[1].scatter(mesh.reshape(-1, 3)[:, 0], mesh.reshape(-1, 3)[:, 1], mesh.reshape(-1, 3)[:, 2], 
                               c=Q_HO_SRCNN[idx, ..., 0].reshape(-1), cmap='jet')
    scatter3 = axes[2].scatter(mesh.reshape(-1, 3)[:, 0], mesh.reshape(-1, 3)[:, 1], mesh.reshape(-1, 3)[:, 2], 
                               c=Q_HO_SRGAN[idx, ..., 0].reshape(-1), cmap='jet')

    # Set labels and titles for each subplot
    for i, ax in enumerate(axes):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(['Original', 'SRCNN', 'SRGAN'][i])

    for i, (scatter, title) in enumerate(zip([scatter1, scatter2, scatter3], ['Original', 'SRCNN', 'SRGAN'])):
        cbar = fig.colorbar(scatter, ax=axes[i], label=r'$\rho u$', shrink=0.6, aspect=10)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])
        axes[i].set_title(title)

    # Adjust layout and display the plot
    plt.tight_layout()

    file_path = f"NEURALNET/figures/fullFlowField_{idx}.png"
    try:
        # Save the figure as a PNG file
        plt.savefig(file_path, format='png')
        plt.close(fig)  # Close all figures to clear memory

        print(f"Figure {idx} saved to '{file_path}' successfully!")
    except Exception as e:
        print(f"Error occurred while saving Figure {idx}: {e}")
        plt.close(fig)

    return

def create_gif_from_images(input_dir, output_path, file_pattern='fullFlowField_*.png', duration=0.5):
    """
    Create an animated GIF from a series of PNG images saved in a directory.

    Parameters:
        input_dir (str): Path to the directory containing PNG images.
        output_path (str): Path to save the animated GIF.
        file_pattern (str): File pattern to match PNG files (default: 'fullFlowField_*.png').
        duration (float): Duration (in seconds) of each frame in the GIF (default: 0.5 seconds).
    """
    try:
        # Get a list of sorted PNG files in the input directory based on file pattern
        file_pattern = f'{input_dir}/{file_pattern}'
        
        file_paths = sorted(glob.glob(file_pattern))
        # Read PNG images and create animated GIF
        images = []
        for file_path in file_paths:
            images.append(imageio.imread(file_path))

        # Write images to GIF file
        imageio.mimsave(output_path, images, duration=duration)

        print(f"Animated GIF saved: {output_path}")

    except Exception as e:
        print(f"Error occurred while creating GIF: {e}")







