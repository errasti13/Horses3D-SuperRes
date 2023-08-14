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
from scipy.interpolate import interpn
import matplotlib.pyplot as plt

class storage():
    Q = []
    
class NNConfig:
    def __init__(self, simulation, n_layers, n_epochs, batch_size, nmin_lo, nmax_lo, nskip_lo,
                 nmin_ho, nmax_ho, nskip_ho, train_percentage, equations, lo_polynomial, ho_polynomial, trained_model):
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

    return NNConfig(
        simulation, n_layers, n_epochs, batch_size, nmin_lo, nmax_lo, nskip_lo,
        nmin_ho, nmax_ho, nskip_ho, train_percentage, equations, lo_polynomial, ho_polynomial, trained_model
    )
def Q_from_file_2(fname):  
    v1 = np.fromfile(fname, dtype=np.int32, count=2, sep='', offset=136)
    
    No_of_elements = v1[0]
    Iter = v1[1]
    
    time = np.fromfile(fname, dtype=np.float64, count=1, sep='', offset=144)
    
    ref_values = np.fromfile(fname, dtype=np.float64, count=6, sep='', offset=152)
    
    Mesh = []
    
    offset_value = 152+6*8+4
    
    
      
    for i in range(0,No_of_elements):

        Local_storage = storage     

        # offset_value = offset_value + 4
        # P_order = np.fromfile(fname, dtype=np.int32, count=4, sep='', offset=offset_value) #208
        # offset_value = offset_value + 4*4   
        # size = P_order[0]*P_order[1]*P_order[2]*P_order[3]
        

            
        # Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
            
        
        
        offset_value = offset_value + 4
        P_order = np.fromfile(fname, dtype=np.int32, count=4, sep='', offset=offset_value) #208
        offset_value = offset_value + 4*4   
        size = P_order[0]*P_order[1]*P_order[2]*P_order[3]   
        
        Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        
        offset_value = offset_value + size*8
        #Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        
        if (i==0):
            Sol = np.zeros((No_of_elements,P_order[0],P_order[1],P_order[2],P_order[3] ))
        else:
            continue
        
        Sol[i,:,:,:,:] =  np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order[0],P_order[1],P_order[2],P_order[3],order='F')
        
   
    return Sol


def Q_from_file(fname):
    v1 = np.fromfile(fname, dtype=np.int32, count=2, sep='', offset=136)
    #print(v1)
    
    No_of_elements = v1[0]
    #print(No_of_elements)
    Iter = v1[1]
    
    time = np.fromfile(fname, dtype=np.float64, count=1, sep='', offset=144)
    
    ref_values = np.fromfile(fname, dtype=np.float64, count=6, sep='', offset=152)
    
    Mesh = []
    
    offset_value = 152+6*8+4
    

   
    for i in range(0,No_of_elements):
        
        #print(i)
        Ind = 0
        
        Local_storage = storage     

        offset_value = offset_value + 4
        P_order = np.fromfile(fname, dtype=np.int32, count=4, sep='', offset=offset_value) #208
        offset_value = offset_value + 4*4   
        size = P_order[0]*P_order[1]*P_order[2]*P_order[3]
        

            
        Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        
        Q1 = np.zeros( (Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3] )   )
        size1 = P_order[0]*P_order[1]*P_order[2]*P_order[3] 
        Q1 =Q

        if (i==0):
            Sol = np.zeros((No_of_elements,P_order[0],P_order[1],P_order[2],P_order[3]))
            #Sol = np.zeros((No_of_elements,size1))
        else:
            Ind = 0
        #print("Q:")
        #print(Q)
        #Local_storage.Q = np.fromfile(fname, dtype=np.float64, count=size , sep='', offset=offset_value).reshape(P_order,order='F')
        Local_storage.Q = Q1.reshape(size1,order='F')
        #print("L Q:")
        #print(Local_storage.Q)
        # if (i==1):
        #     print("Aqui",Q1[1,0,0,0])
        # else:
        #     Ind = 0
        Sol[i,:,:,:,:] = Q1[:,:,:,:]
        #for j in range(0,size1):
        #    Sol[i,j]=Local_storage.Q[j]
        # 'F' en el reshape significa que lo haga como en FORTRAN
        offset_value = offset_value + size*8
        
        #Mesh.append(Local_storage)
    return Sol

def Q_from_experiment(Name,Nmin,Nmax,NSkip):
    Names = []
    
    for i in range(Nmin,Nmax,NSkip):
        s1 = f'{i:010d}'
        Names.append(Name+s1+".hsol") 
       
    N = len(Names)
    Q1   = 0

    for i in range(1,N+1):
        Q1 = Q_from_file(Names[i-1])
        if (i==1):             
            Q = np.zeros((N,Q1.shape[0],Q1.shape[1],Q1.shape[2],Q1.shape[3],Q1.shape[4]) )
            Q[i-1,:,:,:,:,:] = Q1[:,:,:,:,:] 
        else:
            Q[i-1,:,:,:,:,:] = Q1[:,:,:,:,:]  
        
           
    return Q

def Q_SelectEquations(Q_full,Eq):   
 
    Q = np.zeros(( Q_full.shape[0],Q_full.shape[1],len(Eq),Q_full.shape[3],Q_full.shape[4],Q_full.shape[5] ))
    
    for i in range(0,len(Eq)):
        Q[:,:,i,:,:,:] = Q_full[:,:,Eq[i],:,:,:]
        # print(i)
        # for j in range(0,Q_full.shape[0]):
        #     for k in range(0,Q_full.shape[1]):
        #         for l in range(0,Q_full.shape[3]):
        #             for m in range(0,Q_full.shape[4]):
        #                 for n in range(0,Q_full.shape[5]):
        #                     Q[j,k,i,l,m,n] = Q_full[j,k,Eq[i],l,m,n]
    
    return Q



def Read_experiment(Name,Nmin,Nmax,NSkip,Eq,lo_polynomial, Network_type):
    
    Q_full = Q_from_experiment(Name,Nmin,Nmax,NSkip)
    
    
    #print("Qfull",Q_full[1,1,0,0,0,0])
    
    Q = Q_SelectEquations(Q_full,Eq)          # First index: Temporal iteration. 2 index: Number of elements, 3 index Equation index, 4 - 6: x, y, z component

    #print("Qfull 2",Q[1,1,0,0,0,0])
    
    k = 0
    
    N_of_elements = Q.shape[1]
    
    if (Network_type == "MLP"):

        Q_res = np.zeros(( 1, Q.shape[0]*Q.shape[1], Q.shape[2]*Q.shape[3]*Q.shape[4]*Q.shape[5] ))
        for i in range(0,Q.shape[0]):
            for j in range(0,Q.shape[1]):
                Q_res[0,k,:] = Q[i,j,:,:,:].reshape(Q_res.shape[2],order='F')
                k = k + 1



    elif (Network_type == "CNN"):
        
        Q = np.transpose(Q, axes=(0,1,3,4,5,2))

        Q_res = np.zeros((Q.shape[0]*Q.shape[1],lo_polynomial+1,lo_polynomial+1,lo_polynomial+1,3))
        k = 0
        for i in range(0,Q.shape[0]):
            for j in range(0,Q.shape[1]):
                Q_res[k,:,:,:,:] = Q[i,j,:,:,:,:]
                k = k + 1
    else:
        print("Not implemented")
    

    return Q_res, N_of_elements

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


def create_cnn_model(input_shape, num_layers):
    """
    Creates a 3D Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_layers (int): Number of convolutional layers.

    Returns:
        tf.keras.Sequential: The created CNN model.
    """
    model = Sequential()

    model.add(InputLayer(input_shape = input_shape))
    model.add( Conv3D(32, kernel_size=5, dtype = tf.float32, activation = 'relu', kernel_initializer = 'HeUniform', padding = 'same'))
    model.add(BatchNormalization())

    model.add(UpSampling3D(2))
    model.add(Conv3DTranspose(16, (2,2,2), strides=(1, 1, 1)))

    for i in range(num_layers-2):
        model.add(Conv3D(32, kernel_size=3, dtype = tf.float32, kernel_initializer = 'HeUniform', padding = 'same'))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.ReLU())

    model.add(Conv3D(3, kernel_size=1, dtype = tf.float32, activation='linear', kernel_initializer = 'HeUniform'))

    model.summary()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), metrics=['mae'])
    
    return model


def train_model(model, I_train, O_train, I_test, O_test, batch_size, num_epochs):
    """
    Trains a given model using the provided training data.

    Args:
        model (tf.keras.Model): The model to be trained.
        I_train (numpy.ndarray): Input training data.
        O_train (numpy.ndarray): Output training data.
        I_test (numpy.ndarray): Input testing data.
        O_test (numpy.ndarray): Output testing data.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.

    Returns:
        tuple: A tuple containing training and validation loss histories,
               and the history object.
    """
    overfit_callback = EarlyStopping(monitor='loss', patience=8)
    callback = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=10, min_delta=1e-3)

    history = model.fit(
        I_train,
        O_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(I_test, O_test),
        callbacks=[overfit_callback, callback],
    )

    tloss = history.history['loss']
    vloss = history.history['val_loss']

    model.save("NEURALNET/nns/MyModel_SRCNN", save_format='tf')
    model.save("NEURALNET/nns/MyModel_h5_SRCNN", save_format='h5')
    
    return tloss, vloss, history

def calculate_and_print_errors(model, num_iterations, Eq, LO_Polynomial, HO_Polynomial):
    """
    Calculate and print errors for a given model and data.

    Args:
        model (tf.keras.Model): The trained model for prediction.
        GN: Your GN module (assuming it's imported or defined elsewhere).
        num_iterations (int): Number of iterations.
        Eq: Your Eq variable (assuming it's defined elsewhere).
        LO_Polynomial (int): LO polynomial value.
        HO_Polynomial (int): HO polynomial value.
    """
    Q_HO_predict = np.zeros([num_iterations, 512, HO_Polynomial + 1, HO_Polynomial + 1, HO_Polynomial + 1, 3])
    Q_HO_sol = np.zeros([num_iterations, 512, HO_Polynomial + 1, HO_Polynomial + 1, HO_Polynomial + 1, 3])
    L2_Error = []

    for i in range(num_iterations):
        
        print('')
        print("ITERACION", i + 1, "/", num_iterations)

        # Load and preprocess data
        Q_LO_ind, _ = Read_experiment("RESULTS/TGV_LO_", i, i + 1, 1, Eq, LO_Polynomial, "CNN")
        Q_HO_ind, _ = Read_experiment("RESULTS/TGV_HO_", 4 * i, 4 * i + 1, 1, Eq, HO_Polynomial, "CNN")
        I = normalize_and_matrix_4d_predict(Q_LO_ind, 'Q_LO')
        O = normalize_and_matrix_4d_predict(Q_HO_ind, 'Q_HO')

        # Predict and denormalize
        Q_HO_predict[i, :, :, :, :, :] = model.predict(I, verbose=0)
        Q_HO_sol[i, :, :, :, :, :] = denormalize_and_matrix_4d(Q_HO_predict[i, :, :, :, :, :], 'Q_HO')

        # Calculate L2 error
        l2_error = np.linalg.norm(Q_HO_predict[i, :, :, :, :, 0] - O[:, :, :, :, 0]) / np.linalg.norm(O[:, :, :, :, 0])
        L2_Error.append(l2_error)

        # Print information
        print('Normalized min and max values for predicted HO:', np.amax(Q_HO_predict), np.amin(Q_HO_predict))
        print('Normalized min and max values for real HO:', np.amax(O), np.amin(O))
        print("Min and max values for predicted HO:", np.amax(Q_HO_sol), np.amin(Q_HO_sol))
        print("Min and max values for real HO:", np.amax(Q_HO_ind), np.amin(Q_HO_ind))
        print("L2 norm:", l2_error)

    plot_heatmap_and_errors(Q_LO_ind, Q_HO_ind, Q_HO_sol, L2_Error)

    return Q_HO_sol, L2_Error

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

    constant_value = 0.1765
    Interp_Error = np.full(len(L2_Error), constant_value)

    fig = plt.figure()
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.semilogy(0.002 * np.linspace(0, len(L2_Error), len(L2_Error)), L2_Error, label='SRCNN')
    plt.semilogy(0.002 * np.linspace(0, len(L2_Error), len(L2_Error)), Interp_Error, label='Cubic Interpolation')
    plt.legend()
    plt.show()

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
       
    i = 0
    for j in range(8):
        for k in range(8):
            for l in range(8):
                # Extract and assign the coefficients for the corresponding position in the result tensor
                Q_res[l*num_coeffs:(l+1)*num_coeffs, k*num_coeffs:(k+1)*num_coeffs,
                      j*num_coeffs:(j+1)*num_coeffs, :] = Q[i, :, :, :, :]   
                i += 1

    return Q_res

def plot_reconstructed(Q_HO_ind, Q_HO_sol):
    fig, ax = plt.subplots(3, 2)

    im1 = ax[0, 0].matshow(Q_HO_ind[10, :, :,0], cmap='jet')
    im2 = ax[0, 1].matshow(Q_HO_sol[10, :, :, 0], cmap='jet')

    im4 = ax[1, 0].matshow(Q_HO_ind[10, :, :, 1], cmap='jet')
    im5 = ax[1, 1].matshow(Q_HO_sol[10, :, :, 1], cmap='jet')

    im7 = ax[2, 0].matshow(Q_HO_ind[10, :, :, 2], cmap='jet')
    im8 = ax[2, 1].matshow(Q_HO_sol[10, :, :, 2], cmap='jet')

    cbar2 = fig.colorbar(im2, ax=ax[0, :])
    cbar5 = fig.colorbar(im5, ax=ax[1, :])
    cbar8 = fig.colorbar(im8, ax=ax[2, :])

    fig.text(0.1, 0.8, 'Rhou', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.5, 'Rhov', va='center', ha='right', rotation='vertical')
    fig.text(0.1, 0.2, 'Rhow', va='center', ha='right', rotation='vertical')

    ax[0, 0].set_title('Original HO')
    ax[0, 1].set_title('Predicted HO')

    for axes_row in ax:
        for axes in axes_row:
            axes.set_xticks([])
            axes.set_yticks([])

    plt.savefig("NEURALNET/HeatMap_Reconstructed.pdf")
    plt.show()