import numpy as np
from NEURALNET.src.SuperRes_utils import *
from NEURALNET.src.plot import *
from NEURALNET.src.horses3d import *

# Define constants for better readability
HO_FACTOR = 4  # Adjust this according to your use case
LO_INDEX_OFFSET = 2500  # Adjust this according to your use case
HO_INDEX_OFFSET = LO_INDEX_OFFSET*HO_FACTOR  # Adjust this according to your use case

def load_data(variable):
    """Load necessary data files and extract only the specified variable using memory mapping."""

    Q_HO_SRCNN = np.load(f'RESULTS/Q_HO_SRCNN_{variable}.npy', mmap_mode="r")
    Q_HO_SRGAN = np.load(f'RESULTS/Q_HO_SRGAN_{variable}.npy', mmap_mode="r")

    print(f'{variable} has been loaded!')
    return Q_HO_SRCNN, Q_HO_SRGAN


def load_mesh(filename):
    """Load and process the mesh data."""
    return Q_from_file(filename).transpose(0, 2, 3, 4, 1)

def process_and_plot_flow(mesh, Q_HO_SRCNN, Q_HO_SRGAN, config_nn, variable, niter=1000):
    """
    Process and plot flow field data.
    Plots reconstructed flow fields and handles memory exceptions.
    """

    component_map = {'rhou': 1, 'rhov': 2, 'rhow': 3}
    variableIdx = component_map[variable]

    for i in range (niter):
        Q_HO_full = Q_from_file(f"RESULTS/TGV_HO_{config_nn.nmin_ho + i * config_nn.nskip_ho:010d}.hsol").transpose(0, 2, 3, 4, 1)[... ,  variableIdx]
        try:
            plot_reconstructed(mesh, Q_HO_full, Q_HO_SRCNN[i, ...], Q_HO_SRGAN[i, ...], i, component=variable)
            pass
        except MemoryError as e:
            print(f"Memory error occurred at step {i}: {e}")
        del Q_HO_full


def create_animation(input_directory, output_gif_path):
    """Generate a GIF animation from images."""
    create_gif_from_images(input_directory, output_gif_path, file_pattern='fullFlowField_*.png', duration=0.1)

def load_and_plot_error():
    filename_SRCNN = "RESULTS/L2_Error_SRCNN.npy"
    filename_SRGAN = "RESULTS/L2_Error_SRGAN.npy"

    try:
        L2_Error_SRCNN = np.load(filename_SRCNN)
        L2_Error_SRGAN = np.load(filename_SRGAN)
    except Exception as e:
        print(f"Error occurred during file loading: {e}")
        return

    plot_error(L2_Error_SRCNN, L2_Error_SRGAN)


def plot_variable(mesh, config_nn, variable='rhou'):
    rhoU_SRCNN, rhoU_SRGAN = load_data(variable)
    process_and_plot_flow(mesh, rhoU_SRCNN, rhoU_SRGAN, config_nn, variable)
    return


def main():
    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")

    set_equations(config_nn)
    load_and_plot_error()

    mesh_file = "/mnt/c/Users/Jon/Desktop/MUMI/TFM/horses3d_Fernando/horses3d/Solver/test/NavierStokes/TaylorGreenGANS/MESH/TGV_HO_0000000000.hmesh"
    mesh = load_mesh(mesh_file)

    plot_variable(mesh, config_nn, variable='rhou')
    plot_variable(mesh, config_nn, variable='rhov')
    plot_variable(mesh, config_nn, variable='rhow')

    input_directory = "/mnt/c/Users/Jon/Desktop/MUMI/TFM/horses3d_Fernando/horses3d/Solver/test/NavierStokes/TaylorGreenGANS/NEURALNET/figures/"
    output_gif_path = input_directory + "output_animation.gif"
    create_animation(input_directory, output_gif_path)

if __name__ == "__main__":
    main()

