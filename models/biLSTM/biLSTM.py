import tensorflow as tf
import numpy as np
from keras.layers import (
    Layer, Dense,
    Dropout, LayerNormalization,
    LSTM, Bidirectional
)


class BiLSTMEncoderBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(BiLSTMEncoderBlock, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units, return_sequences=True))
        self.projection = tf.keras.layers.Dense(units * 2)  # Match BiLSTM output size
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.bi_lstm(inputs)
        x = self.dropout(x, training=training)
        projected_inputs = self.projection(inputs)  # Project inputs to match x's dimension
        return self.layer_norm(projected_inputs + x)  # Residual connection


class BiLSTMDecoderBlock(Layer):
    def __init__(self, units, dropout_rate=0.1):
        super(BiLSTMDecoderBlock, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units, return_sequences=True))
        self.projection = tf.keras.layers.Dense(units * 2)  # Match BiLSTM output size
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()

    def call(self, inputs, training=False):
        x = self.bi_lstm(inputs)
        x = self.dropout(x, training=training)
        projected_inputs = self.projection(inputs)  # Project inputs to match x's dimension
        return self.layer_norm(projected_inputs + x)  # Residual connection


class LatentBottleneck(Layer):
    def __init__(self, latent_dim):
        super(LatentBottleneck, self).__init__()
        self.latent_dense = Dense(latent_dim)

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1)  # (batch_size, input_dim)
        return self.latent_dense(x)  # (batch_size, latent_dim)


def build_encoder(seq_len, input_dim, latent_dim, lstm_units, num_layers, dropout_rate):
    inputs = tf.keras.Input(shape=(seq_len, input_dim))

    for _ in range(num_layers):
        x = BiLSTMEncoderBlock(units=lstm_units, dropout_rate=dropout_rate)(inputs)

    outputs = LatentBottleneck(latent_dim=latent_dim)(x)  # Latent representation
    return tf.keras.Model(inputs, outputs)


def build_decoder(seq_len, latent_dim, lstm_units, num_layers, output_dim, dropout_rate):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    repeated_latent = tf.keras.layers.RepeatVector(seq_len)(latent_inputs)

    for _ in range(num_layers):
        x = BiLSTMDecoderBlock(units=lstm_units, dropout_rate=dropout_rate)(repeated_latent)

    outputs = Dense(output_dim)(x)  # Reconstructed sequence
    return tf.keras.Model(latent_inputs, outputs)


def build_bilstm_autoencoder(seq_len=96, input_dim=46,
                             latent_dim=46, lstm_units=64,
                             num_layers=3, output_dim=46,
                             dropout_rate=0.1, learning_rate=1e-3):
    # Encoder
    encoder = build_encoder(seq_len=seq_len,
                            input_dim=input_dim,
                            latent_dim=latent_dim,
                            lstm_units=lstm_units,
                            num_layers=num_layers,
                            dropout_rate=dropout_rate
                            )

    # Decoder
    decoder = build_decoder(seq_len=seq_len,
                            latent_dim=latent_dim,
                            lstm_units=lstm_units,
                            num_layers=num_layers,
                            output_dim=output_dim,
                            dropout_rate=dropout_rate
                            )

    # Model
    encoder_inputs = tf.keras.Input(shape=(seq_len, input_dim))
    latent_repr = encoder(encoder_inputs)  # Latent representation
    reconstructed_seq = decoder(latent_repr)  # Reconstructed seq

    model = tf.keras.Model(inputs=encoder_inputs, outputs=reconstructed_seq, name='BiLSTM-AE')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    return model

"""
# Model parameters
SEQ_LEN = 96
INPUT_DIM = 46
LATENT_DIM = 46
LSTM_UNITS = 96
NUM_LAYERS = 3
OUTPUT_DIM = 46

# Input data
X_train = np.random.uniform(-1, 1, size=(800, 96, 46)).astype(np.float32)
X_val = np.random.uniform(-1, 1, size=(200, 96, 46)).astype(np.float32)

# Create model
bilstm_autoencoder = build_bilstm_autoencoder(
    seq_len=SEQ_LEN,
    input_dim=INPUT_DIM,
    latent_dim=LATENT_DIM,
    lstm_units=LSTM_UNITS,
    num_layers=NUM_LAYERS,
    output_dim=OUTPUT_DIM,
)

# Model structure
bilstm_autoencoder.summary()

# Train
bilstm_autoencoder.fit(X_train, X_train, batch_size=32, epochs=3, validation_data=(X_val, X_val))
"""
