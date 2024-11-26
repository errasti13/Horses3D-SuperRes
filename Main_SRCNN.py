import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from NEURALNET.src.SuperRes_utils import *
from NEURALNET.src.horses3d import *
from NEURALNET.src.cnn import *
from NEURALNET.src.gan import *

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected, running on CPU.")

def solve_horses3d(config_nn):
    if config_nn.simulation == 'False':
        print("SOLVING HO SOLUTION")
        os.system('./horses3d.ns TaylorGreen_HO.control > RESULTS/TGV_HO.log')
        print("HO HAS BEEN SOLVED, SOLVING LO SOLUTION")
        os.system('./horses3d.ns TaylorGreen_LO.control > RESULTS/TGV_LO.log')
        print("LO SOLUTION HAS BEEN SOLVED. HORSES3D IS NO LONGER NEEDED.")

    return


def load_and_normalize_data(config_nn, Eq):
    Q_HO_ind, _ = Read_experiment(
        "RESULTS/TGV_HO_", config_nn.nmin_ho, config_nn.nmax_ho, config_nn.nskip_ho, Eq, config_nn.ho_polynomial, "CNN"
    )
    Q_LO_ind, _ = Read_experiment(
        "RESULTS/TGV_LO_", config_nn.nmin_lo, config_nn.nmax_lo, config_nn.nskip_lo, Eq, config_nn.lo_polynomial, "CNN"
    )
    I, MaxValues_LO, MinValues_LO, _, _ = normalize_and_matrix_4d(Q_LO_ind)
    O, MaxValues_FO, MinValues_FO, _, _ = normalize_and_matrix_4d(Q_HO_ind)

    np.save("NEURALNET/MaxValues_LO.npy", MaxValues_LO)
    np.save("NEURALNET/MinValues_LO.npy", MinValues_LO)
    np.save("NEURALNET/MaxValues_FO.npy", MaxValues_FO)
    np.save("NEURALNET/MinValues_FO.npy", MinValues_FO)

    return I, O


def train_model(selected_architecture, I_train, O_train, I_test, O_test, config_nn):
    if selected_architecture == 'SRCNN':
        model = create_cnn_model(I_train.shape[1:], config_nn.n_layers)
        training_history = train_cnn_model(
            model, I_train, O_train, I_test, O_test, config_nn.batch_size, config_nn.n_epochs
        )
        return
    elif selected_architecture == 'SRGAN':
        srgan, generator, discriminator = create_srgan_model(I_train.shape[1:], config_nn.n_layers)
        training_history = train_srgan(
            srgan, generator, discriminator, I_train, O_train, config_nn.batch_size, config_nn.n_epochs, I_test, O_test
        )
        return
    else:
        raise ValueError("Invalid architecture selected.")
    


def load_pretrained_model(selected_architecture):
    if selected_architecture == 'SRCNN':
        return tf.keras.models.load_model("NEURALNET/nns/MyModel_SRCNN")
    elif selected_architecture == 'SRGAN':
        return tf.keras.models.load_model("NEURALNET/nns/MyModel_SRGAN")
    else:
        raise ValueError("Invalid architecture selected.")


def calculate_and_save_results(model, selected_architecture, num_iterations, Eq, config_nn):
    Q_HO_sol, L2_Error = calculate_and_print_errors(model, num_iterations, Eq, config_nn)
    results_prefix = "SRCNN" if selected_architecture == 'SRCNN' else "SRGAN"
    np.save(f"RESULTS/Q_HO_{results_prefix}.npy", Q_HO_sol)
    np.save(f"RESULTS/L2_Error_{results_prefix}.npy", L2_Error)

    return


def main():
    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")
    selected_architecture = config_nn.architecture

    print("#####################################################")
    print(f"##        {selected_architecture} Method has been selected           ##")
    print("#####################################################")

    solve_horses3d(config_nn)
    Eq = set_equations(config_nn)

    if config_nn.trained_model == 'False':
        I, O = load_and_normalize_data(config_nn, Eq)
        I_train, O_train, I_test, O_test = split_train_test_data(I, O, config_nn.train_percentage)

        train_model(selected_architecture, I_train, O_train, I_test, O_test, config_nn)

        del I, O, I_train, O_train, I_test, O_test
    
    model = load_pretrained_model(selected_architecture)

    calculate_and_save_results(model, selected_architecture, num_iterations=400, Eq=Eq, config_nn=config_nn)

    return


if __name__ == "__main__":
    main()
