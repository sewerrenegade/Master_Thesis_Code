import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
import sys
import os
from configs.global_config import GlobalConfig
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy import ImageSequenceClip

sys.path.append("C:/Users\MiladBassil/Desktop/Master_Thesis/code\Master_Thesis_Code")

from models.topology_models.custom_topo_tools.connectivity_topo_regularizer import (
    TopologicalZeroOrderLoss,
)


# Define the custom loss function
class DistanceMatrixLoss(nn.Module):
    def forward(self, original_distances, projected_distances):
        # Mean Squared Error between the distance matrices
        return torch.mean((original_distances - projected_distances) ** 2)


# Downprojection tool
class ConnectivityDP:
    def __init__(
        self,
        n_components=2,
        n_iter=100,
        learning_rate=1,
        optimizer_name = "sgd",
        normalize_input=False,
        initialization_scheme="random_uniform",
        weight_decay=0.0,
        loss_calculation_timeout=1.0,
        augmentation_scheme={},
        importance_weighting = False,
        take_top_p_scales = 1,
        dev_settings={},
    ):
        self.n_iter = n_iter
        self.n_components = n_components
        self.initialization_scheme = initialization_scheme
        self.learning_rate = learning_rate
        self.normalize_input = normalize_input
        self.loss_calculation_timeout = loss_calculation_timeout
        self.take_top_p_scales = take_top_p_scales
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.augmentation_scheme = augmentation_scheme
        self.importance_weighting = importance_weighting
        self.dev_settings = dev_settings
        method = "deep"
        if "moor_method" in self.dev_settings:
            method = "moor_method"

        self.loss_fn = TopologicalZeroOrderLoss(
            method=method, timeout=self.loss_calculation_timeout, take_top_p_scales=self.take_top_p_scales,importance_weighting= importance_weighting
        )
        
    def fit_transform(self, X):
        """
        Downproject the input nxd space to 2D by minimizing the distance matrix loss.
        Args:
            X: Input tensor of shape (n, d).
        Returns:
            Optimized 2D embedding of shape (n, n_componenets).
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if self.normalize_input:
            X = self.normalize_perfeature_input(X)
        initial_embedding = self.get_initial_embedding(X)
        original_distances = self.calculate_eucl_distance_matrix(X)
        target_embedding = initial_embedding

        optimizer = self.get_optimizer(target_embedding)

        progress_bar = tqdm(range(self.n_iter), desc="CDP Progress", unit="step")
        if "create_vid" in self.dev_settings:
            self.create_update_video(initial_embedding, torch.tensor(-1.0), {}, 0)
        for i in progress_bar:
            optimizer.zero_grad()
            projected_distances = self.calculate_eucl_distance_matrix(target_embedding)
            aug_og_distance_matrix = self.augment_distance_matirx(original_distances)
            loss, log = self.loss_fn(aug_og_distance_matrix, projected_distances)
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            if "create_vid" in self.dev_settings:
                self.create_update_video(target_embedding, loss, log, i + 1)
            progress_bar.set_postfix(
                {
                    "Loss": loss.item(),
                    "%_calc": log.get("percentage_toporeg_calc_2_on_1", 100.0),
                } 
            )
        if "create_vid" in self.dev_settings:
            self.create_update_video(None, None, None, -1)
        return target_embedding.detach().numpy()
    
    def get_initial_embedding(self, X):
        np.random.seed(42)
        if self.initialization_scheme == "PCA":
            pca = PCA(n_components=self.n_components)
            return torch.tensor(
                pca.fit_transform(X.numpy()), dtype=torch.float32, requires_grad=True
            )
        elif self.initialization_scheme == "random_uniform":
            return torch.tensor(
                np.random.uniform(-1, 1, size=(X.shape[0], self.n_components)),
                dtype=torch.float32,
                requires_grad=True,
            )
        elif self.initialization_scheme == "random_gaussian":
            return torch.tensor(
                np.random.normal(loc=0, scale=1, size=(X.shape[0], self.n_components)),
                dtype=torch.float32,
                requires_grad=True,
            )
        else:
            raise ValueError(
                f"The initialization method {self.initialization_scheme} is not supported"
            )

    def calculate_eucl_distance_matrix(self, X):
        # Compute the pairwise distance matrix
        return torch.norm(X[:, None] - X, dim=2, p=2)

    def augment_distance_matirx(self, input_distance_matrix):
        if "name" in self.augmentation_scheme:
            if self.augmentation_scheme["name"] == "uniform":
                high = self.augmentation_scheme["p"] + 1
                low = 1 - self.augmentation_scheme["p"]
                upper_triangle = (
                    torch.rand(input_distance_matrix.shape) * (high - low) + low
                )
                symmetric_matrix = (
                    torch.triu(upper_triangle) + torch.triu(upper_triangle, 1).T
                )
                symmetric_matrix.fill_diagonal_(1)
                return input_distance_matrix * symmetric_matrix
        else:
            return input_distance_matrix
        
    def normalize_perfeature_input(self,X):
        return (X - X.mean(dim=1, keepdim=True)) / X.std(dim=1, keepdim=True)

    def get_optimizer(self,opt_domain):
        if self.optimizer_name == "SGD" or self.optimizer_name == "sgd":
            return torch.optim.SGD(
                [opt_domain], lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adam" or self.optimizer_name == "ADAM":
            return torch.optim.Adam(
                [opt_domain], lr=self.learning_rate, weight_decay=self.weight_decay
            )
    

    def create_update_video(self, target_embedding, loss, log, it):
        if it == -1:
            fps = 10
            frames = sorted(
                [
                    os.path.join(self.vid_folder_path, f)
                    for f in os.listdir(self.vid_folder_path)
                    if f.endswith(".png")
                ]
            )
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(
                f"{self.vid_folder_path}vid.mp4", codec="libx264", fps=fps
            )
            return
        if not hasattr(self, "vid_folder_path"):
            from datetime import datetime

            current_time = datetime.now()
            formatted_time = current_time.strftime("%m_%d_%H_%M")
            self.vid_folder_path = (
                GlobalConfig.RESULTS_FOLDER_PATH
                + GlobalConfig.CONNECTIVITY_DP_VID_PATH
                + f"{formatted_time}/"
            )
            if os.path.exists(self.vid_folder_path):
                print(
                    f"WARNING: overwriting old data in folder: {self.vid_folder_path}"
                )
            else:
                os.makedirs(self.vid_folder_path)

        plt.figure(figsize=(8, 8))
        embedding = target_embedding.detach().cpu().numpy()
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=self.dev_settings["labels"],
            cmap="tab10",
            s=50,
        )
        plt.title(f"Iteration {it}, Loss: {loss.item():.4f}")
        plt.savefig(f"{self.vid_folder_path}/frame_{it:05d}.png")
        plt.close()


    def write_iteration_data(file_path, iteration, embedding, loss, histogram, heatmap_stats):
        """Append iteration data to a JSON file in JSON Lines format."""
        # Construct the data to save (ensure minimal size)
        data = {
            "iteration": iteration,
            "embedding": [round(float(e), 4) for e in embedding],  # truncate precision
            "loss": round(float(loss), 6),
            "histogram": histogram,  # e.g., bins or summary stats
            "heatmap": heatmap_stats  # e.g., min, max, mean values
        }
        
        # Append to the JSON Lines file
        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n") 
            
    def read_iteration_data(file_path):
        """Read data line-by-line from a JSON Lines file."""
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)  # Deserialize line into a dictionary
                yield data

# # Example usage
# for iteration_data in read_iteration_data("iterations.jsonl"):
#     print(f"Iteration {iteration_data['iteration']} - Loss: {iteration_data['loss']}")
# Example Usage
if __name__ == "__main__":
    # Generate random input data (nxd)
    n, d = 100, 10
    X = np.random.rand(n, d)

    # Create and fit the downprojection tool
    tool = ConnectivityDP(n_iter=100, learning_rate=0.01)
    embedding = tool.fit_transform(X)

    # Plot the result (requires matplotlib)
    import matplotlib.pyplot as plt

    plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8)
    plt.title("Downprojected Embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
