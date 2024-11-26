import numpy as np
from NEURALNET.src.SuperRes_utils import *
from NEURALNET.src.plot import *
from NEURALNET.src.horses3d import *

<<<<<<< HEAD

HO_FACTOR = 4  # High-order scaling factor
LO_INDEX_OFFSET = 2500  # Low-order index offset
HO_INDEX_OFFSET = LO_INDEX_OFFSET * HO_FACTOR  # High-order index offset
=======
# Define constants for better readability
HO_FACTOR = 4  # Adjust this according to your use case
LO_INDEX_OFFSET = 2500  # Adjust this according to your use case
HO_INDEX_OFFSET = LO_INDEX_OFFSET*HO_FACTOR  # Adjust this according to your use case
>>>>>>> 7b6c70484ef6cf62cee1fc38d337897eba6be43c


<<<<<<< HEAD
def load_data():
    """Load necessary data files."""
=======
    # Load data from saved filescd li   
>>>>>>> 7b6c70484ef6cf62cee1fc38d337897eba6be43c
    Q_HO_SRCNN = np.load("RESULTS/Q_HO_SRCNN.npy")
    Q_HO_SRGAN = np.load("RESULTS/Q_HO_SRGAN.npy")
    L2_Error_SRCNN = np.load("RESULTS/L2_Error_SRCNN.npy")
    L2_Error_SRGAN = np.load("RESULTS/L2_Error_SRGAN.npy")
    return Q_HO_SRCNN, Q_HO_SRGAN, L2_Error_SRCNN, L2_Error_SRGAN


def load_mesh(filename):
    """Load and process the mesh data."""
    return Q_from_file(filename).transpose(0, 2, 3, 4, 1)

<<<<<<< HEAD

def process_and_plot_flow(mesh, Q_HO_SRCNN, Q_HO_SRGAN, config_nn):
    """
    Process and plot flow field data.
    Plots reconstructed flow fields and handles memory exceptions.
    """
    for i in range(400):
        file_path = f"RESULTS/TGV_HO_00000{config_nn.nmin_ho + i * config_nn.nskip_ho}.hsol"
        Q_HO_full = Q_from_file(file_path).transpose(0, 2, 3, 4, 1)
        try:
            plot_reconstructed(mesh, Q_HO_full[..., 2:5], Q_HO_SRCNN, Q_HO_SRGAN, i)
=======
    for i in range (400):
        Q_HO_full = Q_from_file("RESULTS/TGV_HO_00000" + str(config_nn.nmin_ho + i*config_nn.nskip_ho) + ".hsol").transpose(0, 2, 3, 4, 1)
        try:
            plot_reconstructed(mesh, Q_HO_full[..., 1:4], Q_HO_SRCNN, Q_HO_SRGAN, i, component='rhou')
            plot_reconstructed(mesh, Q_HO_full[..., 1:4], Q_HO_SRCNN, Q_HO_SRGAN, i, component='rhov')
            plot_reconstructed(mesh, Q_HO_full[..., 1:4], Q_HO_SRCNN, Q_HO_SRGAN, i, component='rhow')
            pass
>>>>>>> 7b6c70484ef6cf62cee1fc38d337897eba6be43c
        except MemoryError as e:
            print(f"Memory error occurred at step {i}: {e}")
        del Q_HO_full


def create_animation(input_directory, output_gif_path):
    """Generate a GIF animation from images."""
    create_gif_from_images(input_directory, output_gif_path, file_pattern='fullFlowField_*.png', duration=0.1)


def main():
    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")

    # Set up equations using the configuration
    set_equations(config_nn)

    Q_HO_SRCNN, Q_HO_SRGAN, L2_Error_SRCNN, L2_Error_SRGAN = load_data()
    mesh_file = "/mnt/c/Users/Jon/Desktop/MUMI/TFM/horses3d_Fernando/horses3d/Solver/test/NavierStokes/TaylorGreenGANS/MESH/TGV_HO_0000010000.hmesh"
    mesh = load_mesh(mesh_file)

    process_and_plot_flow(mesh, Q_HO_SRCNN, Q_HO_SRGAN, config_nn)

    plot_error(L2_Error_SRCNN, L2_Error_SRGAN)

<<<<<<< HEAD
    input_directory = "/mnt/c/Users/Jon/Desktop/MUMI/TFM/horses3d_Fernando/horses3d/Solver/test/NavierStokes/TaylorGreenGANS/NEURALNET/figures/"
    output_gif_path = input_directory + "output_animation.gif"
    create_animation(input_directory, output_gif_path)

=======
    pass
>>>>>>> 7b6c70484ef6cf62cee1fc38d337897eba6be43c

if __name__ == "__main__":
    main()

