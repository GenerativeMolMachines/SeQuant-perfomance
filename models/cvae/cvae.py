import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from keras.layers import (Input, Dense, Conv2D, AveragePooling2D, BatchNormalization, Flatten, Reshape, Conv2DTranspose,
                          LeakyReLU, Activation, Dropout, Lambda)
from keras.models import Model


def get_filter_count(strategy: str, height: int, i: int, depth: int) -> int:
    if strategy == 'inverse_height':
        if i != depth:
            filter_number = int(height * ((depth - i) / 2))
        else:
            filter_number = 1
        return filter_number
    else:
        raise ValueError(f"Unsupported filter strategy: {strategy}")


def encoder(x: KerasTensor, height: int, width: int, depth: int, filter_strategy: str, noise: bool = False,
        noise_factor: float = 0.1) -> tuple[KerasTensor, list]:
    if noise:
        stddev = tf.abs(x) * noise_factor
        noise_tensor = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, seed=42)
        x = tf.add(x, noise_tensor)

    n_values = []
    for i in range(1, depth + 1):
        if i == 1:
            n = width // 2 ** (depth - 1)
        else:
            n = 2
        n_values.append(n)
        filters = get_filter_count(filter_strategy, height, i, depth)
        x = Conv2D(filters, (1, 4), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D((1, n))(x)
        x = Dropout(0.1)(x)
    return x, n_values


def latent_space(x: tf.Tensor, latent_dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    filters = x.shape[3]
    x = Flatten()(x)

    # Mean value and standard deviation
    mu = Dense(latent_dim, name='mu')(x)
    log_var = Dense(latent_dim, name='log_var')(x)

    # Reparameterization trick
    def sampling(args):
        mu, log_var = args
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    z = Lambda(sampling, name='Latent')([mu, log_var])
    z = Reshape((latent_dim, 1, filters))(z)

    return z, mu, log_var


def decoder(z: tf.Tensor, height: int, depth: int, n_values: list, filter_strategy: str) -> tf.Tensor:
    for i in range(1, depth + 1):
        n = n_values[-i]
        count_list = list(number for number in range(1, 7))
        filters = get_filter_count(filter_strategy, height, count_list[-i], depth)
        z = Conv2DTranspose(filters, (1, 4), strides=(1, n), padding='same')(z)
        z = BatchNormalization()(z)
        if i == depth:
            z = Activation('tanh')(z)
        else:
            z = LeakyReLU(alpha=0.2)(z)
    return z


def ivae_loss(original_inputs, outputs, mu, log_var, z, alpha=0.5, lambda_=1.0):
    """
        InfoVAE Loss Function:
        - Reconstruction loss: MSE
        - Regularization: Weighted KL divergence + MMD term
    """
    # Reconstruction loss (MSE)
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(original_inputs, outputs))

    # KL divergence (weighted by alpha)
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
    kl_loss = alpha * tf.reduce_mean(kl_loss)  # Mean over batch

    # Maximum Mean Discrepancy (MMD)
    def mmd_loss(z):
        """
        Computes the Maximum Mean Discrepancy (MMD) between the latent distribution and prior Gaussian.
        """
        z_prior = tf.random.normal(tf.shape(z))  # Prior distribution (Gaussian)

        def kernel(x, y):
            """RBF Kernel."""
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]

            x = tf.reshape(x, [x_size, 1, dim])
            y = tf.reshape(y, [1, y_size, dim])

            return tf.exp(-tf.reduce_sum(tf.square(x - y), axis=2) / 2)

        k_zz = kernel(z, z)
        k_pp = kernel(z_prior, z_prior)
        k_zp = kernel(z, z_prior)

        mmd = tf.reduce_mean(k_zz) + tf.reduce_mean(k_pp) - 2 * tf.reduce_mean(k_zp)
        return mmd

    mmd = mmd_loss(z)

    # Total loss
    total_loss = reconstruction_loss + kl_loss + lambda_ * mmd
    return total_loss, reconstruction_loss, kl_loss, mmd


def autoencoder_model(height: int, width: int, depth: int, channels: int, latent_dim: int, learning_rate: float,
        filter_strategy: str, noise: bool = False, noise_factor: float = 0.1, alpha: float = 0.5,
        lambda_: float = 1.0) -> tf.keras.Model:
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        inputs = Input(shape=(height, width, channels))
        x = inputs
        x, n_values = encoder(x, height, width, depth, filter_strategy, noise, noise_factor)
        z, mu, log_var = latent_space(x, latent_dim)
        outputs = decoder(z, height, depth, n_values, filter_strategy)

        infovae = Model(inputs, outputs, name='InfoVAE')

        # Compute the losses
        total_loss, reconstruction_loss, kl_loss, mmd = ivae_loss(inputs, outputs, mu, log_var, z, alpha=alpha,
            lambda_=lambda_)

        # Add custom loss
        infovae.add_loss(total_loss)

        # Add metrics for monitoring
        infovae.add_metric(reconstruction_loss, name="reconstruction_loss", aggregation="mean")
        infovae.add_metric(kl_loss, name="kl_loss", aggregation="mean")
        infovae.add_metric(mmd, name="mmd", aggregation="mean")

        # Compile the model
        infovae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    return infovae
