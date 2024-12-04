import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

from Desktop.Master_Thesis_Code.models.topology_models.custom_topo_tools.connectivity_topo_regularizer import TopologicalZeroOrderLoss

# Define the custom loss function
class DistanceMatrixLoss(nn.Module):
    def forward(self, original_distances, projected_distances):
        # Mean Squared Error between the distance matrices
        return torch.mean((original_distances - projected_distances) ** 2)

# Downprojection tool
class DownprojectionTool:
    def __init__(self, n_iter=100, learning_rate=0.01):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.loss_fn = TopologicalZeroOrderLoss(method="deep",timeout=1)

    def calculate_distance_matrix(self, X):
        # Compute the pairwise distance matrix
        return torch.norm(X[:, None] - X, dim=2, p=2)

    def fit(self, X):
        """
        Downproject the input nxd space to 2D by minimizing the distance matrix loss.
        Args:
            X: Input tensor of shape (n, d).
        Returns:
            Optimized 2D embedding of shape (n, 2).
        """
        if isinstance(X,np.ndarray):
            X =  torch.tensor(X, dtype=torch.float32)
        # Step 1: PCA initialization for 2D embedding
        pca = PCA(n_components=2)
        initial_embedding = torch.tensor(pca.fit_transform(X.numpy()), dtype=torch.float32, requires_grad=True)

        # Original distance matrix
        original_distances = self.calculate_distance_matrix(X)

        # Optimizer for the embedding
        optimizer = torch.optim.Adam([embedding], lr=self.learning_rate)
        target_embedding = initial_embedding
        # Step 3: Optimize the 2D embedding
        for i in range(self.n_iter):
            optimizer.zero_grad()

            # Projected distance matrix
            projected_distances = self.calculate_distance_matrix(target_embedding)

            # Calculate loss
            loss = self.loss_fn(original_distances, projected_distances)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{self.n_iter}, Loss: {loss.item()}")

        return embedding.detach().numpy()

# Example Usage
if __name__ == "__main__":
    # Generate random input data (nxd)
    n, d = 100, 10
    X = np.random.rand(n, d)

    # Create and fit the downprojection tool
    tool = DownprojectionTool(n_iter=100, learning_rate=0.01)
    embedding = tool.fit(X)

    # Plot the result (requires matplotlib)
    import matplotlib.pyplot as plt

    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)
    plt.title("Downprojected Embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
