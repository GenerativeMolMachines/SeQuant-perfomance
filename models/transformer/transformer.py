import numpy as np
import tensorflow as tf
from keras import layers, Model, Input


class TransformerEncoderBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)


class LatentBottleneck(layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dense = layers.Dense(latent_dim, name='latent_vec')

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1)

        return self.dense(x)


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, seq_len=96, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, rate=dropout) for _ in range(num_layers)]
        self.pos_emb = TokenAndPositionEmbedding(seq_len, embed_dim)
        self.final_dense = layers.Dense(46, activation=None)

    def call(self, z):
        batch_size = tf.shape(z)[0]
        x = tf.tile(tf.expand_dims(z, axis=1), [1, self.seq_len, 1])
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_dense(x)

        return x


class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_embeddings = self.pos_emb(positions)

        return x + pos_embeddings


def build_encoder(seq_len=96, input_dim=46, embed_dim=128, num_heads=4, ff_dim=256, num_layers=4, dropout=0.1,
                  name='TransformerEncoder'):
    inputs = Input(shape=(seq_len, input_dim), name='encoder_input')
    x = layers.Dense(embed_dim)(inputs)
    x = TokenAndPositionEmbedding(seq_len, embed_dim)(x)
    for i in range(num_layers):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, rate=dropout)(x)

    return Model(inputs, x, name=name)


def latent_bottleneck(encoder_output, latent_dim):
    return LatentBottleneck(latent_dim)(encoder_output)


def build_transformer_autoencoder(seq_len=96, input_dim=46, embed_dim=128, latent_dim=32, num_heads=4, ff_dim=256,
                                  num_layers=4, dropout=0.1, learning_rate=1e-3):
    encoder_model = build_encoder(seq_len=seq_len, input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
                                  ff_dim=ff_dim, num_layers=num_layers, dropout=dropout)

    encoder_inputs = Input(shape=(seq_len, input_dim), name='model_input')
    encoder_output = encoder_model(encoder_inputs)
    z = latent_bottleneck(encoder_output, latent_dim)

    decoder = TransformerDecoder(seq_len=seq_len, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                                 num_layers=num_layers, dropout=dropout)

    reconstructed = decoder(z)

    model = Model(inputs=encoder_inputs, outputs=reconstructed, name='Transmorpher-AE')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    return model


"""
X_train = np.random.uniform(-1, 1, size=(800, 96, 46)).astype(np.float32)
X_val = np.random.uniform(-1, 1, size=(200, 96, 46)).astype(np.float32)

model = build_transformer_autoencoder(
    seq_len=96,  # limitation on sequence length
    input_dim=46,  # input descriptors
    embed_dim=46,  # embeddings dim from attention layers (don't change!)
    latent_dim=46,
    num_heads=4,  # can be tuned
    ff_dim=128,  # can be tuned
    num_layers=4,  # can be tuned
    dropout=0.1,  # can be tuned
    learning_rate=1e-3  # learning rate decreasing can be implemented
)

model.summary()

model.fit(X_train, X_train, batch_size=32, epochs=3, validation_data=(X_val, X_val))
"""
