import numpy as np
from SRCNN_utils import *

# Define constants for better readability
HO_FACTOR = 4  # Adjust this according to your use case
LO_INDEX_OFFSET = 50  # Adjust this according to your use case
HO_INDEX_OFFSET = LO_INDEX_OFFSET*HO_FACTOR  # Adjust this according to your use case

def main():
    # Read the neural network configuration from a file
    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")

    # Set up equations using the configuration
    Eq = set_equations(config_nn)

    # Load data from saved filescd li   
    Q_HO_SRCNN = np.load("RESULTS/Q_HO_SRCNN.npy")
    L2_Error = np.load("RESULTS/L2_Error.npy")

    # Read and process experimental data for high-order reconstruction
    Q_HO_ind, N_of_elements = Read_experiment("RESULTS/TGV_HO_", HO_INDEX_OFFSET, HO_INDEX_OFFSET + 1,
                                               config_nn.nskip_ho, Eq, config_nn.ho_polynomial, "CNN")
    Q_LO_ind, N_of_elements = Read_experiment("RESULTS/TGV_LO_", LO_INDEX_OFFSET, LO_INDEX_OFFSET + 1,
                                               config_nn.nskip_ho, Eq, config_nn.lo_polynomial, "CNN")

    # Reconstruct the fields using the provided function
    Q_HO_ind_reconstruct = reconstruct_field(Q_HO_ind, config_nn.ho_polynomial)
    Q_LO_ind_reconstruct = reconstruct_field(Q_LO_ind, config_nn.lo_polynomial)
    Q_HO_SRCNN_reconstruct = reconstruct_field(Q_HO_SRCNN[LO_INDEX_OFFSET, :, :, :, :, :], config_nn.ho_polynomial)

    # Visualize the reconstructed data (Assuming this function is defined)
    plot_reconstructed(Q_LO_ind_reconstruct, Q_HO_ind_reconstruct, Q_HO_SRCNN_reconstruct)

if __name__ == "__main__":
    main()
