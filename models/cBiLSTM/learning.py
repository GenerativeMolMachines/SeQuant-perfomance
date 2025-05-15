import numpy as np
import os
import pandas as pd
import pickle
import sys
import tensorflow as tf
import time

from preset_tools import (create_dataset_from_batches, oversampling)
from biLSTM_contrastive import *

# variables
monomer_dict = {'A': 'CC(N)C(=O)O', 'R': 'NC(N)=NCCCC(N)C(=O)O', 'N': 'NC(=O)CC(N)C(=O)O', 'D': 'NC(CC(=O)O)C(=O)O',
    'C': 'NC(CS)C(=O)O', 'Q': 'NC(=O)CCC(N)C(=O)O', 'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O',
    'H': 'NC(Cc1cnc[nH]1)C(=O)O', 'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1', 'S': 'NC(CO)C(=O)O',
    'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O', 'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O',
    'O': 'CC1CC=NC1C(=O)NCCCCC(N)C(=O)O', 'U': 'NC(C[Se])C(=O)O'}

# Hyperparameters
max_len = 96
input_dim = 46
latent_dim = 46
lstm_units = 96
num_layers = 3
output_dim = 46
dropout_rate = 0.1

learning_rate = 1e-3
epochs = 30
batch_size = 320

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Training data import
train_df = pd.read_csv('../../data/learning/small_train_df.csv')
test_df = pd.read_csv('../../data/learning/small_test_df.csv')

print('Data has been imported\n')

train_data = list(zip(train_df['sequence'], train_df['cluster']))
test_data = list(zip(test_df['sequence'], test_df['cluster']))

# Oversampling
train_data = oversampling(sequences_with_labels=train_data, target_divisor=batch_size)
test_data = oversampling(sequences_with_labels=test_data, target_divisor=batch_size)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Batching
train_batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
test_batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]

# tf.data.Dataset creation
train_dataset = create_dataset_from_batches(batches=train_batches, monomer_dict=monomer_dict, max_len=max_len)
test_dataset = create_dataset_from_batches(batches=test_batches, monomer_dict=monomer_dict, max_len=max_len)

print('Data preprocessing has been done\n')

# Model initiation
autoencoder = build_bilstm_autoencoder(seq_len=max_len, input_dim=input_dim, latent_dim=latent_dim,
    lstm_units=lstm_units, num_layers=num_layers, output_dim=output_dim, dropout_rate=dropout_rate,
    learning_rate=learning_rate)

print('Start biLSTM model training\n')

# Set contrastive learning step
custom_train_step = CustomTrainStepWithPredefinedLabels(encoder=autoencoder.encoder, decoder=autoencoder.decoder,
                                                        learning_rate=learning_rate)

# Variables due to custom training step
checkpoint_filepath = '../checkpoint/checkpoint_biLSTM_contrastive'
best_val_loss = float('inf')  # For min val loss
patience = 3  # patience for EarlyStopping
wait = 0  # Counter of epochs without improvements

train_loss_history = []
val_loss_history = []

start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_train_loss = []
    epoch_val_loss = []

    # --- Training dataset ---
    for step, (x_batch_train, y_batch_labels) in enumerate(train_dataset):

        losses = custom_train_step.__call__((x_batch_train, y_batch_labels))
        epoch_train_loss.append(losses["loss"])

        if step % 1000 == 0:
            print(f"Step {step}, Loss: {losses['loss']:.4f}, "
                  f"Reconstruction Loss: {losses['reconstruction_loss']:.4f}, "
                  f"Contrastive Loss: {losses['contrastive_loss']:.4f}")

    # Save mean loss per epoch
    train_loss_history.append(tf.reduce_mean(epoch_train_loss))

    # --- Test dataset ---
    for x_batch_val, y_batch_val_labels in test_dataset:
        latent_repr_val = autoencoder.encoder(x_batch_val)
        reconstructed_seq_val = autoencoder.decoder(latent_repr_val)

        reconstruction_loss_val = tf.reduce_mean(tf.keras.losses.mse(x_batch_val, reconstructed_seq_val))
        contrastive_loss_val = custom_train_step.compute_contrastive_loss(latent_repr_val, y_batch_val_labels)

        total_loss_val = reconstruction_loss_val + contrastive_loss_val
        epoch_val_loss.append(total_loss_val)

    val_loss_history.append(tf.reduce_mean(epoch_val_loss))

    # Log results
    print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss_history[-1]:.4f}, "
          f"Val Loss: {val_loss_history[-1]:.4f}")

    # --- Checkpoint ---
    current_val_loss = val_loss_history[-1]
    if current_val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}. Saving model...")
        best_val_loss = current_val_loss
        autoencoder.save(checkpoint_filepath)
        wait = 0
    else:
        wait += 1
        print(f"No improvement in validation loss for {wait} epochs.")

    # --- Early Stopping ---
    if wait >= patience:
        print("Early stopping triggered.")
        break

# Save train history
with open(f'../trainHistoryDict/biLSTM_contrastive.pkl', 'wb') as file_pi:
    history = {"train_loss": train_loss_history, "val_loss": val_loss_history, }
    pickle.dump(history, file_pi)

print("--- %s seconds ---" % (time.time() - start_time))
print()

print(autoencoder.summary())
