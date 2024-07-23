import torch.nn as nn

class RandomProjectionModel(nn.Module):
    """Conv architecture (initially for CIFAR)."""
    def __init__(self, input_dim=[3,32,32]):
        super().__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.name ="Random Convolutions Distance"
        self.projection = nn.Sequential(
            nn.Conv2d(input_dim[0], 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4] ~768 dim
            nn.ReLU(),
            )
    def forward(self, x):
        """Compute latent representation using convolutional net."""
        batch_dim = x.shape[0]
        x = self.projection(x)
        return x.view(batch_dim, -1)

    def get_settings(self):
        return {'input_dim': self.input_dim
            }