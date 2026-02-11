import numpy as np
import matplotlib.pyplot as plt

# Create sample data
X = np.array([
    [20, 20000],
    [22, 25000],
    [25, 30000],
    [45, 120000],
    [47, 130000],
    [50, 150000]
])

# plt.scatter(X[:,0], X[:,1])
# plt.xlabel("Age")
# plt.ylabel("Income")
# plt.title("Raw Data (Unscaled)")
# plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# plt.scatter(X_scaled[:,0], X_scaled[:,1])
# plt.xlabel("Age (scaled)")
# plt.ylabel("Income (scaled)")
# plt.title("Scaled Data")
# plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print(labels)


plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=labels
)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker='X',
    s=200
)
plt.xlabel("Age (scaled)")
plt.ylabel("Income (scaled)")
plt.title("K-Means Clustering (K=2)")
plt.show()


inertia = []

for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 7), inertia, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()



from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=2,
    linkage='ward'
)

labels_h = agg.fit_predict(X_scaled)

print(labels_h)


plt.scatter(
    X_scaled[:,0],
    X_scaled[:,1],
    c=labels_h
)
plt.xlabel("Age (scaled)")
plt.ylabel("Income (scaled)")
plt.title("Hierarchical Clustering")
plt.show()
