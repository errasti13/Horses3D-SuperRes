import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    BatchNormalization, Conv3D, Conv3DTranspose, InputLayer, ReLU,
    Input, UpSampling3D, Flatten, Dense, Reshape, MaxPooling3D, Dropout
)
from keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

from NEURALNET.SRCNN_utils import *

# Set TensorFlow to use GPU memory growth (if available)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def main():

    config_nn = read_nn_config_file("NEURALNET/config_nn.dat")
    selected_architecture = config_nn.architecture

    print("#####################################################")
    print(f"##        {selected_architecture} Method has been selected           ##")
    print("#####################################################")

    if config_nn.simulation == 'False':
        print("SOLVING HO SOLUTION")
        os.system('./horses3d.ns TaylorGreen_HO.control > RESULTS/TGV_HO.log')
        print("HO HAS BEEN SOLVED, SOLVING LO SOLUTION")

        os.system('./horses3d.ns TaylorGreen_lO.control > RESULTS/TGV_lO.log')
        print("LO SOLUTION HAS BEEN SOLVED. HORSES3D IS NO LONGER NEEDED.")
    else:
        pass

    Eq = set_equations(config_nn)

    if config_nn.trained_model == 'False':
        # Data loading and preprocessing
        Q_HO_ind, N_of_elements = Read_experiment("RESULTS/TGV_HO_", config_nn.nmin_ho, config_nn.nmax_ho, config_nn.nskip_ho, Eq, config_nn.ho_polynomial, "CNN")
        
        Q_LO_ind, N_of_elements = Read_experiment("RESULTS/TGV_LO_", config_nn.nmin_lo, config_nn.nmax_lo, config_nn.nskip_lo, Eq, config_nn.lo_polynomial, "CNN")
        
        I, MaxValues_LO, MinValues_LO,  A_lo, a_lo = normalize_and_matrix_4d(Q_LO_ind)
        np.save("NEURALNET/MaxValues_LO.npy", MaxValues_LO)
        np.save("NEURALNET/MinValues_LO.npy", MinValues_LO)

        O, MaxValues_FO, MinValues_FO, A_fo, a_fo = normalize_and_matrix_4d(Q_HO_ind)
        np.save("NEURALNET/MaxValues_FO.npy", MaxValues_FO)
        np.save("NEURALNET/MinValues_FO.npy", MinValues_FO)
        
        I_train, O_train, I_test, O_test = split_train_test_data(I, O, config_nn.train_percentage)

        # Model creation and training
        if selected_architecture == 'SRCNN':
            srcnn_model = create_cnn_model(I_train.shape[1:], config_nn.n_layers)
            tloss, vloss, training_history = train_cnn_model(srcnn_model, I_train, O_train, I_test, O_test, 
                                                         config_nn.batch_size, config_nn.n_epochs)
        elif selected_architecture == 'SRGAN':
            srgan, generator, discriminator = create_srgan_model(I_train.shape[1:], config_nn.n_layers)
            training_history = train_srgan(srgan, generator, discriminator, I_train, O_train, config_nn.batch_size, config_nn.n_epochs, I_test, O_test)
        else:
            print("Invalid architecture selected.")
            return

    #Load pre-trained model
    if selected_architecture == 'SRCNN':
        srcnn_model = tf.keras.models.load_model("NEURALNET/nns/MyModel_SRCNN")

        num_iterations = 400
        Q_HO_sol, L2_Error =calculate_and_print_errors(srcnn_model, num_iterations, Eq, config_nn.lo_polynomial, config_nn.ho_polynomial)

        np.save("RESULTS/Q_HO_SRCNN.npy", Q_HO_sol)
        np.save("RESULTS/L2_Error_SRCNN.npy", L2_Error)

    elif selected_architecture == 'SRGAN':
        srgan_model = tf.keras.models.load_model("NEURALNET/nns/MyModel_SRGAN")

        num_iterations = 400
        Q_HO_sol, L2_Error =calculate_and_print_errors(srgan_model, num_iterations, Eq, config_nn.lo_polynomial, config_nn.ho_polynomial)

        np.save("RESULTS/Q_HO_SRCNN.npy", Q_HO_sol)
        np.save("RESULTS/L2_Error_SRCNN.npy", L2_Error)
    else:
        print("Invalid architecture selected.")
        return
        


    

if __name__ == "__main__":
    main()