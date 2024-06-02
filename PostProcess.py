import numpy as np
from NEURALNET.src.SuperRes_utils import *
from NEURALNET.src.plot import *
from NEURALNET.src.horses3d import *

# Define constants for better readability
HO_FACTOR = 4  # Adjust this according to your use case
LO_INDEX_OFFSET = 2500  # Adjust this according to your use case
HO_INDEX_OFFSET = LO_INDEX_OFFSET*HO_FACTOR  # Adjust this according to your use case

def main():
    # Read the neural network configuration from a file
    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")

    # Load data from saved filescd li   
    Q_HO_SRCNN = np.load("RESULTS/Q_HO_SRCNN.npy")
    Q_HO_SRGAN = np.load("RESULTS/Q_HO_SRGAN.npy")

    L2_Error_SRCNN = np.load("RESULTS/L2_Error_SRCNN.npy")
    L2_Error_SRGAN = np.load("RESULTS/L2_Error_SRGAN.npy")

    # Load mesh 
    filename = "/mnt/c/Users/Jon/Desktop/MUMI/TFM/horses3d_Fernando/horses3d/Solver/test/NavierStokes/TaylorGreenGANS/MESH/TGV_HO_0000010000.hmesh"

    mesh = Q_from_file(filename).transpose(0, 2, 3, 4, 1)

    for i in range (400):
        Q_HO_full = Q_from_file("RESULTS/TGV_HO_00000" + str(config_nn.nmin_ho + i*config_nn.nskip_ho) + ".hsol").transpose(0, 2, 3, 4, 1)
        try:
            plot_reconstructed(mesh, Q_HO_full[..., 1:4], Q_HO_SRCNN, Q_HO_SRGAN, i, component='rhou')
            plot_reconstructed(mesh, Q_HO_full[..., 1:4], Q_HO_SRCNN, Q_HO_SRGAN, i, component='rhov')
            plot_reconstructed(mesh, Q_HO_full[..., 1:4], Q_HO_SRCNN, Q_HO_SRGAN, i, component='rhow')
            pass
        except MemoryError as e:
            print(f"Memory error occurred: {e}")
        del Q_HO_full

    plot_error(L2_Error_SRCNN, L2_Error_SRGAN)

    pass

if __name__ == "__main__":
    main()
