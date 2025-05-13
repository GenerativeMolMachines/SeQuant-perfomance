import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time

from autoencoder import autoencoder_model
from preset_tools import (
    create_dataset_from_batches,
    oversampling
)

# variables
max_len = 96
monomer_dict = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1cnc[nH]1)C(=O)O',
    'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
    'S': 'NC(CO)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O', 'O': 'CC1CC=NC1C(=O)NCCCCC(N)C(=O)O',
    'U': 'NC(C[Se])C(=O)O'
}
# hyperparameters
height = 46
width = 96
depth = 6
channels = 1
latent_dim = height
learning_rate = 1e-3
batch_size = 32
epochs = 75
tf.keras.backend.clear_session()
tf.random.set_seed(2022)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Loading balanced datasets
small_test_df = pd.read_csv('data/small_test_df.csv')
small_train_df = pd.read_csv('data/small_train_df.csv')

train_data = list(small_train_df['sequence'])
test_data = list(small_test_df['sequence'])

# Oversampling
train_data = oversampling(sequences=train_data, target_divisor=batch_size)
test_data = oversampling(sequences=test_data, target_divisor=batch_size)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Batching
train_batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
test_batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]

# tf.data.Dataset creation
train_dataset = create_dataset_from_batches(batches=train_batches, monomer_dict=monomer_dict, max_len=max_len)
test_dataset = create_dataset_from_batches(batches=test_batches, monomer_dict=monomer_dict, max_len=max_len)

# Filter strategies
filter_strategy = ['inverse_height']

# Architectures learning
for strategy in filter_strategy:
    # Clear previous model info
    tf.keras.backend.clear_session()

    print(f'Learning of the model with filter strategy: {strategy}')
    # model init
    autoencoder = autoencoder_model(
        height=height,
        width=width,
        depth=depth,
        channels=channels,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        filter_strategy=strategy
    )

    # set checkpoint
    checkpoint_filepath = f'checkpoint/checkpoint_cae'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    start_time = time.time()

    # Training
    history = autoencoder.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        verbose=2,
        callbacks=[early_stop, model_checkpoint_callback]
    )

    with open(f'trainHistoryDict/cae.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # load model learning history
    with open(f'trainHistoryDict/cae.pkl', 'rb') as f:
        learning_history = pickle.load(f)

    print("--- %s seconds ---" % (time.time() - start_time))
    print()

print('Learning has been finished')
