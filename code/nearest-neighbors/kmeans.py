import numpy as np
import plotly.express as px
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        
    def _euclidean_distance(self, a, b):
        """Calculate Euclidean distance between points and centroids"""
        return np.sqrt(np.sum((a - b)**2, axis=1))
    
    def _calculate_sse(self, X, labels):
        """Calculate sum of squared errors (SSE)"""
        centroids = self.centroids[labels]
        return np.sum((X - centroids) ** 2)
    
    def fit(self, X):
        # Random initialization using Forgy method
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iter):
            # Assignment step: nearest centroid using Voronoi partitioning
            distances = np.array([self._euclidean_distance(X, c) for c in self.centroids])
            labels = np.argmin(distances, axis=0)
            
            # Update step: minimize within-cluster variance
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        distances = np.array([self._euclidean_distance(X, c) for c in self.centroids])
        return np.argmin(distances, axis=0)

# Generate sample data
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

# Train and predict
model = KMeans(k=3)
model.fit(X)
labels = model.predict(X)

# Visualize results
fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels.astype(str),
                 title=f"K-Means Clustering (k=3)<br>SSE: {model._calculate_sse(X, labels):.2f}",
                 labels={'x': 'Feature 1', 'y': 'Feature 2'})
fig.add_scatter(x=model.centroids[:, 0], y=model.centroids[:, 1],
                mode='markers', marker=dict(color='black', size=12, symbol='x'),
                name='Centroids')
fig.show()
