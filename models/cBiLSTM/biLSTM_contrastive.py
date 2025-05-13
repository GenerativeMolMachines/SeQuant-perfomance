import numpy as np
import tensorflow as tf
from keras.layers import (
    Layer, Dense,
    Dropout, LayerNormalization,
    LSTM, Bidirectional
)


class BiLSTMEncoderBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(BiLSTMEncoderBlock, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units, return_sequences=True))
        self.projection = Dense(units * 2)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.bi_lstm(inputs)
        x = self.dropout(x, training=training)
        projected_inputs = self.projection(inputs)
        return self.layer_norm(projected_inputs + x)  # Residual connection


class BiLSTMDecoderBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(BiLSTMDecoderBlock, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units, return_sequences=True))
        self.projection = Dense(units * 2)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.bi_lstm(inputs)
        x = self.dropout(x, training=training)
        projected_inputs = self.projection(inputs)
        return self.layer_norm(projected_inputs + x)  # Residual connection


class LatentBottleneck(Layer):
    def __init__(self, latent_dim):
        super(LatentBottleneck, self).__init__()
        self.latent_dense = Dense(latent_dim)

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1)
        return self.latent_dense(x)


def build_encoder(seq_len, input_dim, latent_dim, lstm_units, num_layers, dropout_rate):
    inputs = tf.keras.Input(shape=(seq_len, input_dim))
    x = inputs
    for _ in range(num_layers):
        x = BiLSTMEncoderBlock(units=lstm_units, dropout_rate=dropout_rate)(x)

    outputs = LatentBottleneck(latent_dim=latent_dim)(x)
    return tf.keras.Model(inputs, outputs)


def build_decoder(seq_len, latent_dim, lstm_units, num_layers, output_dim, dropout_rate):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    repeated_latent = tf.keras.layers.RepeatVector(seq_len)(latent_inputs)

    x = repeated_latent
    for _ in range(num_layers):
        x = BiLSTMDecoderBlock(units=lstm_units, dropout_rate=dropout_rate)(x)

    outputs = Dense(output_dim, activation='tanh')(x)
    return tf.keras.Model(latent_inputs, outputs)


def build_bilstm_autoencoder(seq_len=96, input_dim=46,
                             latent_dim=46, lstm_units=64,
                             num_layers=3, output_dim=46,
                             dropout_rate=0.1, learning_rate=1e-3):
    encoder = build_encoder(seq_len=seq_len,
                            input_dim=input_dim,
                            latent_dim=latent_dim,
                            lstm_units=lstm_units,
                            num_layers=num_layers,
                            dropout_rate=dropout_rate
                            )
    decoder = build_decoder(seq_len=seq_len,
                            latent_dim=latent_dim,
                            lstm_units=lstm_units,
                            num_layers=num_layers,
                            output_dim=output_dim,
                            dropout_rate=dropout_rate
                            )

    encoder_inputs = tf.keras.Input(shape=(seq_len, input_dim))
    latent_repr = encoder(encoder_inputs)
    reconstructed_seq = decoder(latent_repr)

    # Define the autoencoder model
    autoencoder_model = tf.keras.Model(inputs=encoder_inputs, outputs=reconstructed_seq, name='BiLSTM-AE')

    # Compile the model with a custom training step
    autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

    # Attach encoder and decoder to the model for later use
    autoencoder_model.encoder = encoder
    autoencoder_model.decoder = decoder

    return autoencoder_model


class CustomTrainStepWithPredefinedLabels:
    def __init__(self, encoder, decoder, learning_rate):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def __call__(self, data):
        x_batch_train, y_batch_labels = data  # y_batch_labels - cluster labels

        with tf.GradientTape() as tape:
            # Forward pass: encode and decode
            latent_repr = self.encoder(x_batch_train)
            reconstructed_seq = self.decoder(latent_repr)

            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(x_batch_train, reconstructed_seq))

            # Compute contrastive loss using provided labels
            contrastive_loss = self.compute_contrastive_loss(latent_repr, y_batch_labels)

            # Total loss
            total_loss = reconstruction_loss + contrastive_loss

        # Compute gradients and update weights
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": total_loss.numpy(), "reconstruction_loss": reconstruction_loss.numpy(),
                "contrastive_loss": contrastive_loss.numpy()}

    def compute_contrastive_loss(self, latent_repr, cluster_labels):
        """Compute contrastive loss using predefined cluster labels."""
        temperature = 0.5
        batch_size = tf.shape(latent_repr)[0]

        # Normalize embeddings
        normalized_embeddings = tf.math.l2_normalize(latent_repr + 1e-10, axis=1)

        # Compute similarity matrix
        similarity_matrix = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)

        # Create mask for positive samples (same cluster)
        cluster_labels = tf.convert_to_tensor(cluster_labels)
        positive_mask = tf.equal(tf.expand_dims(cluster_labels, 1), tf.expand_dims(cluster_labels, 0))

        # Compute logits with temperature scaling
        logits = similarity_matrix / temperature

        # Mask out diagonal (self-similarity) using a mask
        mask = tf.ones_like(logits) - tf.eye(batch_size)
        logits = logits * mask

        # Debugging: Print logits
        # tf.print("Logits:", logits)

        # Compute probabilities for all pairs
        exp_logits = tf.exp(tf.clip_by_value(logits, -10.0, 10.0))
        partition_function = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
        probabilities = exp_logits / partition_function

        # Debugging: Print probabilities
        # tf.print("Probabilities:", probabilities)

        # Extract probabilities for positive pairs
        positive_probabilities = tf.reduce_sum(probabilities * tf.cast(positive_mask, dtype=tf.float32), axis=1)

        # Avoid log(0) by adding a small epsilon
        positive_probabilities = tf.clip_by_value(positive_probabilities, 1e-10, 1.0)

        # Debugging: Print positive probabilities
        # tf.print("Positive Probabilities:", positive_probabilities)
        positive_loss = -tf.math.log(positive_probabilities)

        return tf.reduce_mean(positive_loss)
