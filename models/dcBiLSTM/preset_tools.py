import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import MinMaxScaler


def make_monomer_descriptors(monomer_dict: dict[str, str]) -> pd.DataFrame:
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in monomer_dict.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(get_descriptors.ComputeProperties(molecule)).reshape((-1, num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)

    sc = MinMaxScaler(feature_range=(-1, 1))
    scaled_array = sc.fit_transform(descriptors_set)
    descriptors_set = pd.DataFrame(scaled_array, columns=descriptor_names, index=monomer_dict.keys())

    energy_data = pd.read_csv('../../data/learning/energy_data.csv')
    energy_set = energy_data.set_index("Aminoacid").iloc[:, :]

    energy_names = energy_set.columns

    scaled_energy = sc.fit_transform(energy_set)
    scaled_energy_set = pd.DataFrame(scaled_energy, columns=energy_names, index=monomer_dict.keys())

    all_descriptors = pd.concat([descriptors_set, scaled_energy_set], axis=1)
    return all_descriptors


def seq_to_matrix(sequence: str, descriptors: pd.DataFrame, num: int):
    rows = descriptors.shape[1]
    seq_matrix = np.empty((0, rows), float)  # shape (0,rows)
    for aa in sequence:
        descriptors_array = np.array(descriptors.loc[aa]).reshape((1, rows))  # shape (1,rows)
        seq_matrix = np.append(seq_matrix, descriptors_array, axis=0)
    seq_matrix = seq_matrix.T
    shape = seq_matrix.shape[1]
    if shape < num:
        add_matrix = np.pad(seq_matrix, [(0, 0), (0, num - shape)], mode='constant', constant_values=0)
        return add_matrix  # shape (rows,n)

    return seq_matrix


def encode_seqs(sequences_list: list[str], descriptors: pd.DataFrame, num: int):
    lst = []
    i = 0
    for sequence in sequences_list:
        seq_matrix = seq_to_matrix(sequence=sequence, descriptors=descriptors, num=num)
        lst.append(seq_matrix)
        i += 1
    encoded_seqs = np.dstack(lst)  # (num_features, sequence_length, batch_size)

    return encoded_seqs


def preprocess_input(peptides):
    peptides = peptides.reshape((peptides.shape[0], peptides.shape[1], peptides.shape[2], 1))
    return peptides


def data_processing(batch_data: list[tuple[str, int]],  # tuples(sequence, label)
                    monomer_dict: dict[str, str], max_len: int):
    sequences, labels = zip(*batch_data)  # Split sequences and labels
    descriptors_set = make_monomer_descriptors(monomer_dict)
    batch_encoded_sequences = encode_seqs(sequences, descriptors_set, max_len)

    batch_processed = np.moveaxis(batch_encoded_sequences, [0, 1, 2], [2, 1, 0])

    return batch_processed, np.array(labels)  # Return processed sequences and array with labels


def create_dataset_from_batches(batches: list[list[tuple[str, int]]],
                                # list of batches, each batch - list of tuples (sequence, label)
                                monomer_dict: dict[str, str], max_len: int):
    def generator():
        for batch in batches:
            processed_batch, labels = data_processing(batch, monomer_dict, max_len)
            yield processed_batch, labels  # Return sequences and labels

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # input (sequences)
        tf.TensorSpec(shape=(None,), dtype=tf.int32)  # target (cluster labels)
    ))
    return dataset


def oversampling(sequences_with_labels: list[tuple[str, int]],  # list of tuples (sequence, label)
                 target_divisor: int):
    current_size = len(sequences_with_labels)
    remainder = current_size % target_divisor

    if remainder == 0:
        print("Dataset size is already equal to ", target_divisor)
        return sequences_with_labels

    additional_records_needed = target_divisor - remainder

    sampled_indices = np.random.choice(len(sequences_with_labels), size=additional_records_needed, replace=True)
    oversampled_sequences_with_labels = [sequences_with_labels[i] for i in sampled_indices]

    result_sequences_with_labels = sequences_with_labels + oversampled_sequences_with_labels

    return result_sequences_with_labels
