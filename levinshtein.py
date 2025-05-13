import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.manifold import MDS
from Levenshtein import distance
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

root_data = "./data/embeddings_SeQuant/"
classification_datasets = ['antimic', 'antidia', 'antiinf', 'antiox']
regression_datasets = ['Instability_Index', 'Theoretical_Net_Charge', 'Isoelectric_Point', 'Molecular_Weight']

def _get_positions(sequences: list) -> np.array:
    distances = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            distances[i, j] = distance(sequences[i], sequences[j])
            distances[j, i] = distances[i, j]
    return distances

def mean_dist(data_name):
    data_path = root_data + data_name +"_sequant.csv"
    data = pd.read_csv(data_path)
    sequences = data['seq'].values.tolist()
    distances = _get_positions(sequences)
    print(f"{data_name}: {np.mean(distances)}")
    with open(f'distances_{data_name}.txt', 'w') as f:
        for item in sequences:
            f.write("%s\n" % item)
    return np.mean(distances)

for name in classification_datasets:
    mean_dist(name)

for name in regression_datasets:
    mean_dist(name)
