import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv3D, Conv3DTranspose, InputLayer, ReLU, UpSampling3D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape, num_layers, learning_rate=1e-2):
    """
    Creates a 3D Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_layers (int): Number of convolutional layers.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: The created CNN model.
    """
    model = Sequential(name="3D_CNN_Model")

    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv3D(32, kernel_size=5, activation='relu', kernel_initializer='HeUniform', padding='same', dtype=tf.float32))
    model.add(BatchNormalization())

    model.add(UpSampling3D(size=2))
    model.add(Conv3DTranspose(16, kernel_size=(2, 2, 2), strides=(1, 1, 1)))

    for _ in range(num_layers - 2):
        model.add(Conv3D(32, kernel_size=3, kernel_initializer='HeUniform', padding='same', dtype=tf.float32))
        model.add(BatchNormalization())
        model.add(ReLU())

    model.add(Conv3D(3, kernel_size=1, activation='linear', kernel_initializer='HeUniform', dtype=tf.float32))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

    model.summary()

    return model


def train_cnn_model(model, I_train, O_train, I_test, O_test, batch_size, num_epochs, save_path="NEURALNET/nns/MyModel_SRCNN"):
    """
    Trains a given CNN model using the provided training data.

    Args:
        model (tf.keras.Model): The model to be trained.
        I_train (numpy.ndarray): Input training data.
        O_train (numpy.ndarray): Output training data.
        I_test (numpy.ndarray): Input testing data.
        O_test (numpy.ndarray): Output testing data.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.
        save_path (str): Path to save the trained model.

    Returns:
        tuple: A tuple containing training loss, validation loss histories,
               and the history object.
    """

    callbacks = [
        EarlyStopping(monitor='loss', patience=80, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.2, patience=100, min_delta=1e-3)
    ]

    history = model.fit(
        I_train,
        O_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(I_test, O_test),
        callbacks=callbacks,
        verbose=1
    )

    model.save(save_path)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    return train_loss, val_loss, history
