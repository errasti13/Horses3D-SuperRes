import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from NEURALNET.src.horses3d import *
from NEURALNET.src.plot import *
    
class NNConfig:
    def __init__(self, simulation, n_layers, n_epochs, batch_size, nmin_lo, nmax_lo, nskip_lo,
                 nmin_ho, nmax_ho, nskip_ho, train_percentage, equations, lo_polynomial, ho_polynomial, trained_model, architecture):
        self.simulation = simulation
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nmin_lo = nmin_lo
        self.nmax_lo = nmax_lo
        self.nskip_lo = nskip_lo
        self.nmin_ho = nmin_ho
        self.nmax_ho = nmax_ho
        self.nskip_ho = nskip_ho
        self.train_percentage = train_percentage
        self.equations = equations
        self.lo_polynomial = lo_polynomial
        self.ho_polynomial = ho_polynomial
        self.trained_model = trained_model
        self.architecture = architecture

def read_nn_config_file(filepath):
    NN_opt = []

    with open(filepath, 'r') as f:
        NN_opt = f.read().splitlines()

    simulation = NN_opt[0].split('\t')[0]
    n_layers = int(NN_opt[1].split('\t')[0])
    n_epochs = int(NN_opt[2].split('\t')[0])
    batch_size = int(NN_opt[3].split('\t')[0])
    nmin_lo = int(NN_opt[4].split('\t')[0])
    nmax_lo = int(NN_opt[5].split('\t')[0])
    nskip_lo = int(NN_opt[6].split('\t')[0])
    nmin_ho = int(NN_opt[7].split('\t')[0])
    nmax_ho = int(NN_opt[8].split('\t')[0])
    nskip_ho = int(NN_opt[9].split('\t')[0])
    train_percentage = float(NN_opt[10].split('\t')[0])
    equations = NN_opt[11].split('\t')[0]
    lo_polynomial = int(NN_opt[12].split('\t')[0])
    ho_polynomial = int(NN_opt[13].split('\t')[0])
    trained_model = NN_opt[14].split('\t')[0]
    architecture = NN_opt[15].split('\t')[0]

    return NNConfig(
        simulation, n_layers, n_epochs, batch_size, nmin_lo, nmax_lo, nskip_lo,
        nmin_ho, nmax_ho, nskip_ho, train_percentage, equations, lo_polynomial, ho_polynomial, trained_model, architecture
    )

def set_equations(config_nn):
    """
    Sets the equation numbers based on the configuration.

    Args:
        config_nn (NNConfig): The configuration object containing the equation information.

    Returns:
        list: A list of equation numbers based on the configuration.
    """
    equation_sets = {
    'momentum': [2, 3, 4],
    'all': [1, 2, 3, 4, 5]
    }

    return equation_sets.get(config_nn.equations, [])

def normalize_and_matrix_4d(Q):

    """
    Normalizes and reshapes a 4D tensor Q.

    Args:
        Q (numpy.ndarray): The input tensor to normalize and reshape.

    Returns:
        numpy.ndarray: The normalized and reshaped tensor.
        numpy.ndarray: Maximum values of the normalized tensor's columns.
        numpy.ndarray: Minimum values of the normalized tensor's columns.
        numpy.ndarray: Scaling matrix for denormalization.
        numpy.ndarray: Translation vector for denormalization.
    """
    Q2d = Q.reshape((Q.shape[0], -1), order='F')

    MaxValues = np.max(Q2d, axis=0)
    MinValues = np.min(Q2d, axis=0)
    MeaValues = 0.5 * (MaxValues + MinValues)
    DelValues = MaxValues - MinValues

    A_lo = np.diag(1 / DelValues)
    a_lo = - MeaValues @ A_lo + 0.5

    Qnew = (Q2d @ A_lo) + a_lo
    Qnew = Qnew.reshape(Q.shape, order='F')

    return Qnew, MaxValues, MinValues, A_lo, a_lo

def denormalize_and_matrix_4d(Q, Name):
    """
    Denormalizes a 4D tensor Q based on the specified normalization parameters.

    Args:
        Q (numpy.ndarray): The input tensor to denormalize.
        Name (str): Specifies whether Q is from LO or HO simulation.

    Returns:
        numpy.ndarray: The denormalized tensor.
    """
    if Name == 'Q_LO':
        MaxValues = np.load("NEURALNET/MaxValues_LO.npy")
        MinValues = np.load("NEURALNET/MinValues_LO.npy")
    elif Name == 'Q_HO':
        MaxValues = np.load("NEURALNET/MaxValues_FO.npy")
        MinValues = np.load("NEURALNET/MinValues_FO.npy")

    DelValues = MaxValues - MinValues

    Q2d = Q.reshape((Q.shape[0], Q.shape[1] * Q.shape[2] * Q.shape[3] * Q.shape[4]), order='F')
    Qnew = Q2d * DelValues + MinValues

    Qnew = Qnew.reshape((Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3], Q.shape[4]), order='F')

    return Qnew

def normalize_and_matrix_4d_predict(Q, Name):
    """
    Normalizes and reshapes a 4D tensor Q based on the specified normalization parameters.

    Args:
        Q (numpy.ndarray): The input tensor to normalize and reshape.
        Name (str): Specifies whether Q is from LO or HO simulation.

    Returns:
        numpy.ndarray: The normalized and reshaped tensor.
    """
    Q2d = Q.reshape((Q.shape[0], Q.shape[1] * Q.shape[2] * Q.shape[3] * Q.shape[4]), order='F')

    N = Q2d.shape[1]
    MaxValues = np.zeros(N)
    MinValues = np.zeros(N)
    MeaValues = np.zeros(N)
    DelValues = np.zeros(N)

    Qnew = np.zeros(Q2d.shape)

    if Name == 'Q_LO':
        MaxValues = np.load("NEURALNET/MaxValues_LO.npy")
        MinValues = np.load("NEURALNET/MinValues_LO.npy")
    elif Name == 'Q_HO':
        MaxValues = np.load("NEURALNET/MaxValues_FO.npy")
        MinValues = np.load("NEURALNET/MinValues_FO.npy")

    for i in range(N):
        MeaValues[i] = 0.5 * (MaxValues[i] + MinValues[i])
        DelValues[i] = MaxValues[i] - MinValues[i]

    a_lo = - MeaValues
    A_lo = np.diag(1 / DelValues)
    a_lo = np.matmul(A_lo, a_lo) + 0.5

    for i in range(Q.shape[0]):
        Qnew[i, :] = np.matmul(A_lo, Q2d[i, :]) + a_lo

    Qnew = Qnew.reshape(Q.shape, order='F')

    return Qnew



def split_train_test_data(I, O, Train_percentage):
    """
    Splits the input and output data into training and testing sets.

    Args:
        I (numpy.ndarray): Input data array.
        O (numpy.ndarray): Output data array.
        Train_percentage (float): Percentage of data to be used for training.

    Returns:
        I_train (numpy.ndarray): Input training data.
        O_train (numpy.ndarray): Output training data.
        I_test (numpy.ndarray): Input testing data.
        O_test (numpy.ndarray): Output testing data.
    """
    Ntrain = int(Train_percentage * I.shape[0])
    Ntest = I.shape[0] - Ntrain
    
    IndexTotal = np.random.randint(0, I.shape[0], I.shape[0])
    IndexTrain = IndexTotal[:Ntrain]
    IndexTest = IndexTotal[Ntrain:]
    
    I_train = I[IndexTrain]
    O_train = O[IndexTrain]
    I_test = I[IndexTest]
    O_test = O[IndexTest]
    
    return I_train, O_train, I_test, O_test


def calculate_and_print_errors(model, num_iterations, Eq, config_nn):
    """
    Calculate and print errors for a given model and data.

    Args:
        model (tf.keras.Model): The trained model for prediction.
        num_iterations (int): Number of iterations.
        Eq: Equation variable for the experiment (assumed defined externally).
        config_nn: Configuration object containing hyperparameters.
    Returns:
        np.ndarray: High-order solution array.
        list: L2 error values for each iteration.
    """
    Q_HO_sol = np.zeros(
        (num_iterations, 512, config_nn.ho_polynomial + 1, config_nn.ho_polynomial + 1, config_nn.ho_polynomial + 1, 3),
        dtype=np.float32
    )
    L2_Error = []

    for i in range(num_iterations):
        print(f'\nIteration {i + 1}/{num_iterations}')

        lo_start = config_nn.nmin_lo + i * config_nn.nskip_lo
        lo_end = config_nn.nmin_lo + (i + 1) * config_nn.nskip_lo
        ho_start = config_nn.nmin_ho + i * config_nn.nskip_ho
        ho_end = config_nn.nmin_ho + (i + 1) * config_nn.nskip_ho

        Q_LO_ind, _ = Read_experiment("RESULTS/TGV_LO_", lo_start, lo_end, 1, Eq, config_nn.lo_polynomial, "CNN")
        Q_HO_ind, _ = Read_experiment("RESULTS/TGV_HO_", ho_start, ho_end, config_nn.nskip_ho, Eq, config_nn.ho_polynomial, "CNN")

        I = normalize_and_matrix_4d_predict(Q_LO_ind, 'Q_LO')
        O = normalize_and_matrix_4d_predict(Q_HO_ind, 'Q_HO')

        Q_HO_predict = model.predict(I, verbose=0)
        Q_HO_sol[i] = denormalize_and_matrix_4d(Q_HO_predict, 'Q_HO')

        l2_error = np.linalg.norm(Q_HO_predict[..., 0] - O[..., 0]) / np.linalg.norm(O[..., 0])
        L2_Error.append(l2_error)

        print(f'L2 norm error: {l2_error:.6f}')
        print(f'Predicted HO (normalized) range: {Q_HO_predict.min():.6f}, {Q_HO_predict.max():.6f}')
        print(f'Real HO (normalized) range: {O.min():.6f}, {O.max():.6f}')
        print(f'Predicted HO (denormalized) range: {Q_HO_sol[i].min():.6f}, {Q_HO_sol[i].max():.6f}')
        print(f'Real HO (raw) range: {Q_HO_ind.min():.6f}, {Q_HO_ind.max():.6f}')

    return Q_HO_sol, L2_Error



def reconstruct_field(Q, ho_polynomial):
    """
    Reconstructs a field using higher-order polynomial expansion.

    Args:
        Q (numpy.ndarray): Input tensor of shape (N, 512, M, M, M, 3), where N is the number of samples.
        ho_polynomial (int): Degree of the higher-order polynomial.

    Returns:
        numpy.ndarray: Reconstructed field tensor of shape (N, 8*(ho_polynomial+1), 8*(ho_polynomial+1), 8*(ho_polynomial+1), 3).
    """
    # Calculate the number of coefficients in each dimension
    num_coeffs = ho_polynomial + 1

    # Define the shape of the reconstructed field
    field_shape = (8 * num_coeffs, 8 * num_coeffs, 8 * num_coeffs, 3)
    
    # Initialize the result tensor
    Q_res = np.zeros(field_shape)
   
    eID = 0
    for x in range(8):
        for y in range(8):
                for z in range(8):
                    # Assign the corresponding segment of Q to Q_res
                    element = np.transpose(Q[eID, :, :, :, :], (2, 1, 0, 3))
                    Q_res[x * (ho_polynomial+1):(x+1) * (ho_polynomial + 1), 
                        y * (ho_polynomial+1):(y+1) * (ho_polynomial + 1) , 
                        z * (ho_polynomial+1):(z+1) * (ho_polynomial + 1), 
                        :] = element
                    eID += 1

    return Q_res
