import tensorflow as tf
from keras.layers import (
    BatchNormalization, Conv3D, Conv3DTranspose, InputLayer, LeakyReLU,
    Input, UpSampling3D, Flatten, Dense
)
from keras.models import Sequential,  Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import time

def create_srgan_model(input_shape, num_layers):
    """
    Creates a Super-Resolution Generative Adversarial Network (SRGAN) model.

    Args:
        input_shape (tuple): Shape of the input data.
        num_layers (int): Number of convolutional layers for the generator.

    Returns:
        tf.keras.Model: The created SRGAN model.
    """
    # Generator (SRCNN)
    generator = Sequential(name='generator')
    generator.add(InputLayer(input_shape=input_shape))
    generator.add(Conv3D(32, kernel_size=5, dtype=tf.float32, activation='relu', kernel_initializer='he_uniform', padding='same'))
    generator.add(BatchNormalization())

    generator.add(UpSampling3D(size=(2, 2, 2)))
    generator.add(Conv3DTranspose(16, kernel_size=(2, 2, 2), strides=(1, 1, 1)))

    for _ in range(num_layers - 2):
        generator.add(Conv3D(32, kernel_size=3, dtype=tf.float32, kernel_initializer='he_uniform', padding='same'))
        generator.add(BatchNormalization())
        generator.add(tf.keras.layers.ReLU())

    generator.add(Conv3D(3, kernel_size=1, dtype=tf.float32, activation='linear', kernel_initializer='he_uniform'))

    # Discriminator
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

    # Combined model (Generator + Discriminator)
    input_hr = Input(shape=input_shape)
    generated_hr = generator(input_hr)
    discriminator.trainable = False
    validity = discriminator(generated_hr)

    srgan = Model(input_hr, [validity, generated_hr], name='srgan')
    
    # Compile discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=2e-3), metrics=['accuracy'])

    # Compile SRGAN
    srgan.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 1], optimizer=Adam(learning_rate=1e-2))
    srgan.summary()
    
    return srgan, generator, discriminator

def train_srgan(srgan, generator, discriminator, I_train, O_train, batch_size, epochs, I_test, O_test):
    tf.keras.utils.disable_interactive_logging()

    """
    Trains the Super-Resolution Generative Adversarial Network (SRGAN).

    Args:
        srgan (tf.keras.Model): The SRGAN model.
        discriminator (tf.keras.Model): The discriminator model.
        I_train (numpy.ndarray): Low-resolution training data.
        O_train (numpy.ndarray): High-resolution training data.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        I_test (numpy.ndarray): Low-resolution test data.
        O_test (numpy.ndarray): High-resolution test data.

    Returns:
        History: A tf.keras.callbacks.History object that records training metrics.
    """
    train_lr_batches = []
    train_hr_batches = []
    for it in range(int(I_train.shape[0] // batch_size)):
        start_idx = it * batch_size
        end_idx   = (it + 1) * batch_size

        train_lr_batches.append(I_train[start_idx:end_idx])
        train_hr_batches.append(O_train[start_idx:end_idx])

    # Learning rate scheduler
    def lr_scheduler(epoch):
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

        g_losses = []
        d_losses = []

        for d in range(len(train_hr_batches)):
            lr_imgs = train_lr_batches[d]
            hr_imgs = train_hr_batches[d]

            fake_imgs = generator.predict_on_batch(lr_imgs)

            #First, train the discriminator on fake and real HR images. 
            discriminator.trainable = True
            d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)

            #Now, train the generator by fixing discriminator as non-trainable
            discriminator.trainable = False

            #Average the discriminator loss, just for reporting purposes. 
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

            #Train the generator via GAN. 
            #Remember that we have 2 losses, adversarial loss and content (VGG) loss
            g_loss = srgan.train_on_batch(lr_imgs, [real_label, hr_imgs])


            #Save losses to a list so we can average and report. 
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
        #Convert the list of losses to an array to make it easy to average    
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        
        #Calculate the average losses for generator and discriminator
        g_loss = np.sum(g_losses, axis=0) / len(g_losses)
        d_loss = np.sum(d_losses, axis=0) / len(d_losses)

        # Calculate epoch execution time
        epoch_time = time.time() - start_time

        # Print the progress
        print(f"Epoch {epoch+1}/{epochs} - Epoch time: {epoch_time:.2f}s - Learning rate: {lr:.2e} - D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}% - G loss: {g_loss[0]:.4f}")

        # Save losses for plotting
        history.append({'D_loss': d_loss, 'G_loss': g_loss})

    generator.save("NEURALNET/nns/MyModel_SRGAN")
    return history 