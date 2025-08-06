import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# initialize scaler
scaler = StandardScaler()

# explore
penguins_df = pd.read_csv("penguins.csv")
print(penguins_df.head())
print(penguins_df.isna().value_counts())

# drop NA
penguins_df = penguins_df.dropna().reset_index(drop=True)
print(penguins_df.isna().value_counts())

# transform to numeric
penguins_df = pd.get_dummies(penguins_df, drop_first=True, dtype=np.int8)
print(penguins_df)

# choosing k number
k_inertia = []
for k in range(1, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(penguins_df)
    k_inertia.append(kmeans.inertia_)
plt.plot(range(1, 7), k_inertia, marker='o')
plt.xlabel('k')
plt.ylabel('inertia')
plt.title('elbow method')
plt.show()

# scaling
penguins_df_scaled = scaler.fit_transform(penguins_df)

# fit kmeans
k = 2
model = KMeans(n_clusters=k)
model.fit(penguins_df_scaled)
clusters = model.transform(penguins_df_scaled)
labels = model.labels_
print(labels)

# assign labels to data
print(clusters.shape, labels.shape)
labeled_clusters = np.hstack((clusters, labels.reshape(-1, 1)))
labeled_clusters_df = pd.DataFrame(labeled_clusters, columns=['x', 'y', 'cluster'])
print(labeled_clusters_df)

# plot the clusters
plt.scatter(x=labeled_clusters_df['x'], y=labeled_clusters_df['y'], c=labeled_clusters_df['cluster'])
plt.show()

# result for average
penguins_df['cluster'] = labeled_clusters_df['cluster']
stat_penguins = penguins_df.groupby('cluster').aggregate('mean').iloc[:, :4]
print(stat_penguins)

