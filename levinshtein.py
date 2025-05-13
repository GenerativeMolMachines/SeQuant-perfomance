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

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(distances)
    sample = pos[-1]
    pos = pos[:-2, :]

    k = 2
    connectivity = kneighbors_graph(pos, n_neighbors=k, include_self=False)
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage='ward',
        connectivity=connectivity
    )
    model.fit(pos)
    return model, pos, sample

data_name = classification_datasets[0]

data_path = root_data + data_name +"_sequant.csv"
data = pd.read_csv(data_path)
sequences = data['seq'].values.tolist()
model, pos, sample = _get_positions(sequences)

with open(f'pos_{data_name}.txt', 'w') as f:
    for item in sequences:
        f.write("%s\n" % item)

print(np.mean(pos))
