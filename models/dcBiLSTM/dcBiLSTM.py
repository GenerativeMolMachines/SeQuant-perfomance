import numpy as np
import tensorflow as tf
from keras.layers import (
    Layer, Dropout,
    LayerNormalization,
    LSTM, Bidirectional
)
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable()
class DiagonalLSTMInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        n_rows, n_cols = shape
        part_size = n_cols // 4

        matrix = tf.zeros(shape, dtype=dtype)
        for gate in range(4):
            for i in range(min(n_rows, part_size)):
                indices = [[i, gate * part_size + i]]
                updates = [0.1]
                matrix = tf.tensor_scatter_nd_update(matrix, indices, updates)

        return matrix

    def get_config(self):
        return {}


class BiLSTMEncoderBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(BiLSTMEncoderBlock, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units,
            return_sequences=True,
            use_bias=True,
            kernel_initializer=DiagonalLSTMInitializer(),
            recurrent_initializer=DiagonalLSTMInitializer(),
            bias_initializer='zeros'
        ))
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.bi_lstm(inputs)
        x = self.dropout(x, training=training)
        return self.layer_norm(x)


class BiLSTMDecoderBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(BiLSTMDecoderBlock, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units,
            return_sequences=True,
            use_bias=True,
            kernel_initializer=DiagonalLSTMInitializer(),
            recurrent_initializer=DiagonalLSTMInitializer(),
            bias_initializer='zeros'
        ))
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.bi_lstm(inputs)
        x = self.dropout(x, training=training)
        return self.layer_norm(x)


class LatentBottleneck(Layer):
    def __init__(self):
        super(LatentBottleneck, self).__init__()

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1)
        return x


def build_encoder(seq_len, input_dim, latent_dim, lstm_units, num_layers, dropout_rate):
    inputs = tf.keras.Input(shape=(seq_len, input_dim))
    x = inputs
    for _ in range(num_layers):
        x = BiLSTMEncoderBlock(units=lstm_units, dropout_rate=dropout_rate)(x)

    outputs = LatentBottleneck()(x)
    return tf.keras.Model(inputs, outputs)


def build_decoder(seq_len, latent_dim, lstm_units, num_layers, output_dim, dropout_rate):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    repeated_latent = tf.keras.layers.RepeatVector(seq_len)(latent_inputs)

    x = repeated_latent
    for _ in range(num_layers):
        x = BiLSTMDecoderBlock(units=lstm_units, dropout_rate=dropout_rate)(x)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_dim, activation='tanh', use_bias=False)
    )(x)
    return tf.keras.Model(latent_inputs, outputs)


def build_bilstm_autoencoder(seq_len=96, input_dim=46,
                             latent_dim=92, lstm_units=46,
                             num_layers=1, output_dim=46,
                             dropout_rate=0.1, learning_rate=1e-4):
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

            # new losses for non-diag matix
            encoder_diag_loss, encoder_diag_energy_ratio, encoder_off_diag_energy_ratio = self.custom_diag_penalty_get_weights(self.encoder)
            decoder_diag_loss, decoder_diag_energy_ratio, decoder_off_diag_energy_ratio = self.custom_diag_penalty_get_weights(self.decoder)

            # Total loss
            total_loss = reconstruction_loss + contrastive_loss + encoder_diag_loss + decoder_diag_loss

        # Compute gradients and update weights
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "loss": total_loss.numpy(),
            "reconstruction_loss": reconstruction_loss.numpy(),
            "contrastive_loss": contrastive_loss.numpy(),
            "encoder_diag_loss": encoder_diag_loss.numpy(),
            "decoder_diag_loss": decoder_diag_loss.numpy(),
            # Energy metrics monitoring
            "encoder_diag_energy_ratio": encoder_diag_energy_ratio.numpy(),
            "encoder_off_diag_energy_ratio": encoder_off_diag_energy_ratio.numpy(),
            "decoder_diag_energy_ratio": decoder_diag_energy_ratio.numpy(),
            "decoder_off_diag_energy_ratio": decoder_off_diag_energy_ratio.numpy()
        }

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

    def compute_energy_metrics(self, matrix):
        abs_matrix = tf.abs(matrix)
        n_rows = tf.shape(abs_matrix)[0]
        n_cols = tf.shape(abs_matrix)[1]

        diag_mask = tf.eye(n_rows, n_cols)
        off_diag_mask = 1.0 - diag_mask

        total_energy = tf.reduce_sum(tf.square(abs_matrix)) + 1e-10
        diag_energy = tf.reduce_sum(tf.square(abs_matrix * diag_mask))
        off_diag_energy = tf.reduce_sum(tf.square(abs_matrix * off_diag_mask))

        diag_energy_ratio = diag_energy / total_energy
        off_diag_energy_ratio = off_diag_energy / total_energy

        return diag_energy_ratio, off_diag_energy_ratio


    def custom_diag_penalty_get_weights(self, model, penalty_coef=100.0):
        penalty = 0.0
        total_diag_energy = 0.0
        total_off_diag_energy = 0.0
        matrix_count = 0

        def nondiagonal_penalty(matrix):
            abs_matrix = tf.abs(matrix)
            n_rows = tf.shape(abs_matrix)[0]
            n_cols = tf.shape(abs_matrix)[1]

            diag_mask = tf.eye(n_rows, n_cols)
            off_diag_mask = 1.0 - diag_mask

            nondiag = abs_matrix * off_diag_mask
            return tf.reduce_mean(tf.square(nondiag))

        # Search all BiLSTM-blocks
        for layer in model.layers:
            if hasattr(layer, "bi_lstm"):
                bi_lstm = layer.bi_lstm
                for lstm in [bi_lstm.forward_layer, bi_lstm.backward_layer]:
                    # Use trainable_variables instead of get_weights()
                    kernel, recurrent_kernel, _ = lstm.cell.trainable_weights # In case of use_bias=False - delete third variable

                    W_i, W_f, W_c, W_o = tf.split(kernel, 4, axis=1)
                    U_i, U_f, U_c, U_o = tf.split(recurrent_kernel, 4, axis=1)

                    for W_mat in [W_i, W_f, W_c, W_o]:
                        penalty += nondiagonal_penalty(W_mat)
                        diag_ratio, off_diag_ratio = self.compute_energy_metrics(W_mat)
                        total_diag_energy += diag_ratio
                        total_off_diag_energy += off_diag_ratio
                        matrix_count += 1

                    for U_mat in [U_i, U_f, U_c, U_o]:
                        penalty += nondiagonal_penalty(U_mat)
                        diag_ratio, off_diag_ratio = self.compute_energy_metrics(U_mat)
                        total_diag_energy += diag_ratio
                        total_off_diag_energy += off_diag_ratio
                        matrix_count += 1

        avg_diag_energy = total_diag_energy / matrix_count if matrix_count > 0 else 0.0
        avg_off_diag_energy = total_off_diag_energy / matrix_count if matrix_count > 0 else 1.0

        return penalty_coef * penalty, avg_diag_energy, avg_off_diag_energy
