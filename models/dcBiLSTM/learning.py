import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from preset_tools import (
    create_dataset_from_batches,
    oversampling
)
from dcBiLSTM import *

# variables
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

# Hyperparameters
max_len = 96
input_dim = 46
latent_dim = 92
lstm_units = 46
num_layers = 1
output_dim = 46
dropout_rate = 0.1

learning_rate = 1e-4
epochs = 50
batch_size = 320

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
os.environ["KERAS_BACKEND"] = "tensorflow"

# Training data import
train_df = pd.read_csv('../data/small_train_df_with_clusters_2_mini.csv')
test_df = pd.read_csv('../data/small_test_df_with_clusters_2_mini.csv')

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
autoencoder = build_bilstm_autoencoder(
    seq_len=max_len,
    input_dim=input_dim,
    latent_dim=latent_dim,
    lstm_units=lstm_units,
    num_layers=num_layers,
    output_dim=output_dim,
    dropout_rate=dropout_rate,
    learning_rate=learning_rate
)

# Force weights initialization
dummy_input = tf.random.normal((1, max_len, input_dim))
_ = autoencoder(dummy_input)

print('Start biLSTM model training\n')

# Set contrastive learning step
custom_train_step = CustomTrainStepWithPredefinedLabels(encoder=autoencoder.encoder, decoder=autoencoder.decoder, learning_rate=learning_rate)

print("=== Pre-training Energy Metrics ===")
encoder_diag_loss, encoder_diag_ratio, encoder_off_diag_ratio = custom_train_step.custom_diag_penalty_get_weights(autoencoder.encoder)
decoder_diag_loss, decoder_diag_ratio, decoder_off_diag_ratio = custom_train_step.custom_diag_penalty_get_weights(autoencoder.decoder)

print(f"Encoder - Diag: {encoder_diag_ratio:.3f}, Off-diag: {encoder_off_diag_ratio:.3f}")
print(f"Decoder - Diag: {decoder_diag_ratio:.3f}, Off-diag: {decoder_off_diag_ratio:.3f}")

# Variables due to custom training step
checkpoint_filepath = '../checkpoint/dcBiLSTM'
best_val_loss = float('inf')  # For min val loss
patience = 3
wait = 0  # Counter of epochs without improvements

start_time = time.time()

# Histories (store per-epoch values)
train_total_history = []
train_recon_history = []
train_contrastive_history = []
train_encdiag_history = []
train_decdiag_history = []
# Histories for energy metrics monitoring
train_enc_diag_energy_history = []
train_enc_off_diag_energy_history = []
train_dec_diag_energy_history = []
train_dec_off_diag_energy_history = []

val_total_history = []
val_recon_history = []
val_contrastive_history = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    # Per-epoch accumulators (lists of floats)
    epoch_train_total = []
    epoch_train_recon = []
    epoch_train_contrastive = []
    epoch_train_encdiag = []
    epoch_train_decdiag = []
    # Per-epoch accumulators for energy metrics monitoring
    epoch_train_enc_diag_energy = []
    epoch_train_enc_off_diag_energy = []
    epoch_train_dec_diag_energy = []
    epoch_train_dec_off_diag_energy = []

    # --- Training loop (no per-step printing) ---
    for step, (x_batch_train, y_batch_labels) in enumerate(train_dataset):
        # call the custom train step (it returns dict with numpy/scalar values)
        losses = custom_train_step.__call__((x_batch_train, y_batch_labels))
        # append all returned components
        epoch_train_total.append(float(losses["loss"]))
        epoch_train_recon.append(float(losses["reconstruction_loss"]))
        epoch_train_contrastive.append(float(losses["contrastive_loss"]))
        epoch_train_encdiag.append(float(losses["encoder_diag_loss"]))
        epoch_train_decdiag.append(float(losses["decoder_diag_loss"]))
        # monitoring energy metrics
        epoch_train_enc_diag_energy.append(float(losses["encoder_diag_energy_ratio"]))
        epoch_train_enc_off_diag_energy.append(float(losses["encoder_off_diag_energy_ratio"]))
        epoch_train_dec_diag_energy.append(float(losses["decoder_diag_energy_ratio"]))
        epoch_train_dec_off_diag_energy.append(float(losses["decoder_off_diag_energy_ratio"]))

    # compute per-epoch means (training)
    train_total = float(np.mean(epoch_train_total)) if len(epoch_train_total) > 0 else float('nan')
    train_recon = float(np.mean(epoch_train_recon)) if len(epoch_train_recon) > 0 else float('nan')
    train_contrastive = float(np.mean(epoch_train_contrastive)) if len(epoch_train_contrastive) > 0 else float('nan')
    train_encdiag = float(np.mean(epoch_train_encdiag)) if len(epoch_train_encdiag) > 0 else float('nan')
    train_decdiag = float(np.mean(epoch_train_decdiag)) if len(epoch_train_decdiag) > 0 else float('nan')

    train_enc_diag_energy = float(np.mean(epoch_train_enc_diag_energy)) if len(
        epoch_train_enc_diag_energy) > 0 else float('nan')
    train_enc_off_diag_energy = float(np.mean(epoch_train_enc_off_diag_energy)) if len(
        epoch_train_enc_off_diag_energy) > 0 else float('nan')
    train_dec_diag_energy = float(np.mean(epoch_train_dec_diag_energy)) if len(
        epoch_train_dec_diag_energy) > 0 else float('nan')
    train_dec_off_diag_energy = float(np.mean(epoch_train_dec_off_diag_energy)) if len(
        epoch_train_dec_off_diag_energy) > 0 else float('nan')

    # append to global histories
    train_total_history.append(train_total)
    train_recon_history.append(train_recon)
    train_contrastive_history.append(train_contrastive)
    train_encdiag_history.append(train_encdiag)
    train_decdiag_history.append(train_decdiag)

    train_enc_diag_energy_history.append(train_enc_diag_energy)
    train_enc_off_diag_energy_history.append(train_enc_off_diag_energy)
    train_dec_diag_energy_history.append(train_dec_diag_energy)
    train_dec_off_diag_energy_history.append(train_dec_off_diag_energy)

    # --- Validation loop (compute only recon + contrastive, no diag penalties) ---
    epoch_val_total = []
    epoch_val_recon = []
    epoch_val_contrastive = []

    for x_batch_val, y_batch_val_labels in test_dataset:
        latent_repr_val = autoencoder.encoder(x_batch_val)
        reconstructed_seq_val = autoencoder.decoder(latent_repr_val)

        # compute validation losses (convert to float)
        reconstruction_loss_val = float(tf.reduce_mean(tf.keras.losses.mse(x_batch_val, reconstructed_seq_val)).numpy())
        contrastive_loss_val = float(custom_train_step.compute_contrastive_loss(latent_repr_val, y_batch_val_labels).numpy())

        total_loss_val = reconstruction_loss_val + contrastive_loss_val

        epoch_val_recon.append(reconstruction_loss_val)
        epoch_val_contrastive.append(contrastive_loss_val)
        epoch_val_total.append(total_loss_val)

    # compute per-epoch means (validation)
    val_total = float(np.mean(epoch_val_total)) if len(epoch_val_total) > 0 else float('nan')
    val_recon = float(np.mean(epoch_val_recon)) if len(epoch_val_recon) > 0 else float('nan')
    val_contrastive = float(np.mean(epoch_val_contrastive)) if len(epoch_val_contrastive) > 0 else float('nan')

    val_total_history.append(val_total)
    val_recon_history.append(val_recon)
    val_contrastive_history.append(val_contrastive)

    # --- Per-epoch summary print (single line concise) ---
    print(f"Epoch {epoch + 1} Summary:")
    print(f"  Train Total: {train_total:.4f} | Recon: {train_recon:.4f} | Contrastive: {train_contrastive:.4f} | "
          f"EncDiag: {train_encdiag:.4f} | DecDiag: {train_decdiag:.4f}")
    print(f"  Val   Total: {val_total:.4f} | Recon: {val_recon:.4f} | Contrastive: {val_contrastive:.4f}")
    print(f"  Energy: EncDiag={train_enc_diag_energy:.3f}, EncOff={train_enc_off_diag_energy:.3f}, "
          f"DecDiag={train_dec_diag_energy:.3f}, DecOff={train_dec_off_diag_energy:.3f}")

    # --- Checkpoint logic (same as before) ---
    current_val_loss = val_total
    if current_val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.6f} to {current_val_loss:.6f}. Saving model...")
        best_val_loss = current_val_loss
        autoencoder.save(checkpoint_filepath)
        wait = 0
    else:
        wait += 1
        print(f"No improvement in validation loss for {wait} epochs.")

    if epoch % 3 == 0:
        print("=== Weight Sanity Check ===")
        for i, layer in enumerate(autoencoder.encoder.layers):
            if hasattr(layer, 'bi_lstm'):
                for j, lstm in enumerate([layer.bi_lstm.forward_layer, layer.bi_lstm.backward_layer]):
                    kernel = lstm.cell.kernel.numpy()
                    non_zero_count = np.count_nonzero(kernel)
                    print(f"Layer {i}, LSTM {j}: Non-zero elements: {non_zero_count}")

    # --- Early stopping ---
    if wait >= patience:
        print("Early stopping triggered.")
        break


# Save train history  -------------------------------------------------------
history_to_save = {
    "train_total": train_total_history,
    "train_reconstruction": train_recon_history,
    "train_contrastive": train_contrastive_history,
    "train_encoder_diag": train_encdiag_history,
    "train_decoder_diag": train_decdiag_history,
    "train_encoder_diag_energy": train_enc_diag_energy_history,
    "train_encoder_off_diag_energy": train_enc_off_diag_energy_history,
    "train_decoder_diag_energy": train_dec_diag_energy_history,
    "train_decoder_off_diag_energy": train_dec_off_diag_energy_history,
    "val_total": val_total_history,
    "val_reconstruction": val_recon_history,
    "val_contrastive": val_contrastive_history
}

history_path = f'../trainHistoryDict/biLSTM_contrastive_diag_loss_16_16_1_mini.pkl'
with open(history_path, 'wb') as file_pi:
    pickle.dump(history_to_save, file_pi)

print("--- %s seconds ---" % (time.time() - start_time))
print()

print(autoencoder.summary())
