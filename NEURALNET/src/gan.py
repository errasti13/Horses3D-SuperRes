import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv3D, Conv3DTranspose, InputLayer, LeakyReLU,
    Input, UpSampling3D, Flatten, Dense, ReLU
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import time

def build_generator(num_layers, input_shape):
    generator = Sequential(name='generator')
    generator.add(InputLayer(input_shape=input_shape))
    generator.add(Conv3D(32, kernel_size=5, activation='relu', kernel_initializer='he_uniform', padding='same'))
    generator.add(BatchNormalization())
    generator.add(UpSampling3D(size=(2, 2, 2)))
    generator.add(Conv3DTranspose(16, kernel_size=(2, 2, 2), strides=(1, 1, 1)))

    for _ in range(num_layers - 2):
        generator.add(Conv3D(32, kernel_size=3, kernel_initializer='he_uniform', padding='same'))
        generator.add(BatchNormalization())
        generator.add(ReLU())

    generator.add(Conv3D(3, kernel_size=1, activation='linear', kernel_initializer='he_uniform'))

    return generator


def build_discriminator(generator):
    discriminator = Sequential(name='discriminator')
    discriminator.add(InputLayer(input_shape=generator.output_shape[1:]))
    discriminator.add(Conv3D(32, kernel_size=3, strides=(1, 1, 1), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv3D(32, kernel_size=3, strides=(2, 2, 2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv3D(32, kernel_size=3, strides=(2, 2, 2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))

    return discriminator

def create_srgan_model(input_shape, num_layers, lr_generator=1e-2, lr_discriminator=2e-3):
    """
    Creates a Super-Resolution Generative Adversarial Network (SRGAN) model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_layers (int): Number of convolutional layers for the generator.
        lr_generator (float): Learning rate for the generator.
        lr_discriminator (float): Learning rate for the discriminator.

    Returns:
        tuple: SRGAN model, generator model, and discriminator model.
    """

    generator = build_generator(num_layers, input_shape)

    discriminator = build_discriminator(generator)

    input_hr = Input(shape=input_shape)
    generated_hr = generator(input_hr)
    discriminator.trainable = False
    validity = discriminator(generated_hr)

    srgan = Model(input_hr, [validity, generated_hr], name='srgan')

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=lr_discriminator), metrics=['accuracy'])
    srgan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=Adam(learning_rate=lr_generator))

    srgan.summary()
    return srgan, generator, discriminator


def train_srgan(srgan, generator, discriminator, I_train, O_train, batch_size, epochs, I_test, O_test, save_path="NEURALNET/nns/MyModel_SRGAN"):
    """
    Trains the Super-Resolution Generative Adversarial Network (SRGAN).

    Args:
        srgan (tf.keras.Model): The SRGAN model.
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        I_train (numpy.ndarray): Low-resolution training data.
        O_train (numpy.ndarray): High-resolution training data.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        I_test (numpy.ndarray): Low-resolution test data.
        O_test (numpy.ndarray): High-resolution test data.
        save_path (str): Path to save the trained generator model.

    Returns:
        list: Training history containing generator and discriminator losses.
    """

    train_lr_batches = [I_train[i:i + batch_size] for i in range(0, I_train.shape[0], batch_size)]
    train_hr_batches = [O_train[i:i + batch_size] for i in range(0, O_train.shape[0], batch_size)]

    def lr_scheduler(epoch):
        """Adjusts learning rate every 100 epochs."""
        if epoch % 100 == 0 and epoch != 0:
            lr = tf.keras.backend.get_value(srgan.optimizer.lr)
            tf.keras.backend.set_value(srgan.optimizer.lr, lr * 0.1)
        return tf.keras.backend.get_value(srgan.optimizer.lr)

    history = []
    for epoch in range(epochs):
        start_time = time.time()
        lr = lr_scheduler(epoch)

        fake_label = np.zeros((batch_size, 1))
        real_label = np.ones((batch_size, 1))
        g_losses, d_losses = [], []

        for lr_imgs, hr_imgs in zip(train_lr_batches, train_hr_batches):
            fake_imgs = generator.predict_on_batch(lr_imgs)

            discriminator.trainable = True
            d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            discriminator.trainable = False
            g_loss = srgan.train_on_batch(lr_imgs, [real_label, hr_imgs])

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        d_loss_avg = np.mean(d_losses, axis=0)
        g_loss_avg = np.mean(g_losses, axis=0)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s - LR: {lr:.2e} - "
              f"D loss: {d_loss_avg[0]:.4f}, acc: {100 * d_loss_avg[1]:.2f}% - "
              f"G loss: {g_loss_avg[0]:.4f}")

        history.append({'D_loss': d_loss_avg, 'G_loss': g_loss_avg})

    generator.save(save_path)
    return history
