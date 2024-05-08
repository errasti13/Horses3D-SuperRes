import tensorflow as tf
from keras.layers import (
    BatchNormalization, Conv3D, Conv3DTranspose, InputLayer, ReLU, LeakyReLU,
    Input, UpSampling3D, Flatten, Dense, Reshape, MaxPooling3D, Dropout
)
from keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

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

def train_cnn_model(model, I_train, O_train, I_test, O_test, batch_size, num_epochs):
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
    overfit_callback = EarlyStopping(monitor='loss', patience=80)
    callback = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=100, min_delta=1e-3)

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

    model.save("NEURALNET/nns/MyModel_SRCNN")
    
    return tloss, vloss, history